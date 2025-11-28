# pip install opencv-python-headless
# pip install tifffile
# pip install numpy
# pip install pillow
# pip install scikit-image

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tifffile
import os
import cv2
import warnings
from PIL import Image

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# EFFICIENT LIGHTWEIGHT MODEL FOR SMALL DATASETS
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class EfficientUNet(nn.Module):
    """Lightweight U-Net optimized for small datasets"""

    def __init__(self, n_channels=5, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Reduced channel dimensions for small dataset
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up4 = Up(64, 32)
        self.outc = nn.Conv2d(32, n_classes, 1)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ============================================================================
# PREPROCESSING WITH ADVANCED NORMALIZATION
# ============================================================================

def robust_normalize(img):
    """Robust normalization handling edge cases"""
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    if img.max() == img.min():
        return np.zeros_like(img)

    valid_pixels = img[img > 0]
    if len(valid_pixels) > 0:
        p2, p98 = np.percentile(valid_pixels, [2, 98])
        if p98 - p2 > 1e-6:
            img_norm = np.clip((img - p2) / (p98 - p2), 0, 1)
        else:
            img_norm = np.clip(img / (img.max() + 1e-8), 0, 1)
    else:
        img_norm = np.clip(img / (img.max() + 1e-8), 0, 1)

    return img_norm.astype(np.float32)


def preprocess_image(bands_data, target_size=256):
    """Preprocess multi-band image with robust normalization"""
    processed_bands = []
    for band_img in bands_data:
        if band_img.ndim == 3:
            band_img = band_img[..., 0]
        band_norm = robust_normalize(band_img)
        processed_bands.append(band_norm)

    multi_band = np.stack(processed_bands, axis=-1)

    if multi_band.shape[0] != target_size or multi_band.shape[1] != target_size:
        multi_band = cv2.resize(multi_band, (target_size, target_size),
                                interpolation=cv2.INTER_LINEAR)

    return multi_band


# ============================================================================
# TEST TIME AUGMENTATION (TTA)
# ============================================================================

def tta_predict(model, image, use_tta=True):
    """Test-time augmentation for robust predictions"""
    predictions = []

    with torch.no_grad():
        # Original
        pred = torch.sigmoid(model(image))
        predictions.append(pred)

        if use_tta:
            # Horizontal flip
            flipped_h = torch.flip(image, dims=[3])
            pred_h = torch.sigmoid(model(flipped_h))
            pred_h = torch.flip(pred_h, dims=[3])
            predictions.append(pred_h)

            # Vertical flip
            flipped_v = torch.flip(image, dims=[2])
            pred_v = torch.sigmoid(model(flipped_v))
            pred_v = torch.flip(pred_v, dims=[2])
            predictions.append(pred_v)

            # Both flips
            flipped_hv = torch.flip(image, dims=[2, 3])
            pred_hv = torch.sigmoid(model(flipped_hv))
            pred_hv = torch.flip(pred_hv, dims=[2, 3])
            predictions.append(pred_hv)

    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred


# ============================================================================
# POST-PROCESSING
# ============================================================================

def post_process_mask(mask, min_size=50):
    """Remove small artifacts and smooth boundaries"""
    from skimage import morphology
    from skimage.measure import label, regionprops

    # Convert to binary
    binary_mask = (mask > 0).astype(np.uint8)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Remove small regions
    labeled = label(binary_mask)
    for region in regionprops(labeled):
        if region.area < min_size:
            binary_mask[labeled == region.label] = 0

    return binary_mask


# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================

def get_tile_id(filename):
    """Extract tile ID from filename"""
    import re
    match = re.search(r"img(\d+)\.tif", filename)
    if match:
        return match.group(1)
    match = re.search(r"(\d{2}_\d{2})", filename)
    return match.group(1) if match else filename.replace('.tif', '')


def maskgeration(imagepath, out_dir):
    """
    Generate binary masks for glacier segmentation

    Args:
        imagepath: Dictionary mapping band names to folder paths
        out_dir: Output directory for saving masks
    """
    IMG_SIZE = 256
    USE_TTA = True
    USE_POST_PROCESS = True

    # Check if out_dir is a file (model.pth) - runner.py bug workaround
    if os.path.isfile(out_dir) or out_dir.endswith('.pth'):
        # Extract directory from model path
        out_dir = os.path.dirname(out_dir)
        if not out_dir:
            out_dir = '/work'

    # Create output directory safely
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = EfficientUNet(n_channels=5, n_classes=1).to(device)

    model_paths = [
        'model.pth',
        '/work/model.pth',
        os.path.join(os.getcwd(), 'model.pth')
    ]

    model_loaded = False
    optimal_threshold = 0.5

    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimal_threshold = checkpoint.get('threshold', 0.5)
                else:
                    model.load_state_dict(checkpoint)
                model_loaded = True
                print(f"Model loaded from: {model_path}")
                print(f"Optimal threshold: {optimal_threshold}")
                break
            except Exception as e:
                print(f"Failed to load from {model_path}: {e}")
                continue

    if not model_loaded:
        raise FileNotFoundError("model.pth not found in any expected location")

    model.eval()

    # Build tile mapping
    band_tile_map = {band: {} for band in imagepath}
    for band, folder in imagepath.items():
        if not os.path.exists(folder):
            print(f"Warning: Folder not found - {folder}")
            continue

        files = os.listdir(folder)
        for f in files:
            if f.endswith(".tif"):
                tile_id = get_tile_id(f)
                if tile_id:
                    band_tile_map[band][tile_id] = f

    # Get tile IDs from first band
    ref_band = sorted(imagepath.keys())[0]
    tile_ids = sorted(band_tile_map[ref_band].keys())

    print(f"Processing {len(tile_ids)} tiles...")
    masks_generated = {}

    # Process each tile
    for idx, tile_id in enumerate(tile_ids):
        try:
            bands = []
            H, W = None, None
            output_filename = None

            # Load all bands for this tile
            for band_name in sorted(imagepath.keys()):
                if tile_id not in band_tile_map[band_name]:
                    print(f"Warning: Missing {band_name} for tile {tile_id}")
                    continue

                filename = band_tile_map[band_name][tile_id]
                file_path = os.path.join(imagepath[band_name], filename)

                if not os.path.exists(file_path):
                    print(f"Warning: File not found - {file_path}")
                    continue

                # Read band
                band_image = tifffile.imread(file_path).astype(np.float32)
                if band_image.ndim == 3:
                    band_image = band_image[..., 0]

                if H is None:
                    H, W = band_image.shape
                    output_filename = filename

                bands.append(band_image)

            if len(bands) != 5:
                print(f"Warning: Incomplete bands for tile {tile_id} ({len(bands)}/5)")
                continue

            # Preprocess
            multi_band_image = preprocess_image(bands, target_size=IMG_SIZE)

            # Create tensor
            image_tensor = torch.from_numpy(multi_band_image.transpose(2, 0, 1)).float()
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # Predict
            if USE_TTA:
                prediction = tta_predict(model, image_tensor, use_tta=True)
            else:
                with torch.no_grad():
                    prediction = torch.sigmoid(model(image_tensor))

            # Threshold
            binary_mask = (prediction > optimal_threshold).float()
            binary_mask = binary_mask.squeeze().cpu().numpy()

            # Resize to original dimensions
            if binary_mask.shape != (H, W):
                binary_mask = cv2.resize(binary_mask, (W, H),
                                         interpolation=cv2.INTER_NEAREST)

            # Post-process
            if USE_POST_PROCESS:
                binary_mask = post_process_mask(binary_mask, min_size=30)

            # Convert to uint8
            binary_mask = (binary_mask * 255).astype(np.uint8)

            # Save mask
            output_path = os.path.join(out_dir, output_filename)
            tifffile.imwrite(output_path, binary_mask)

            masks_generated[tile_id] = binary_mask

            if (idx + 1) % 5 == 0:
                print(f"Processed {idx + 1}/{len(tile_ids)} tiles")

        except Exception as e:
            print(f"Error processing tile {tile_id}: {e}")
            continue

    print(f"Completed! Masks saved to: {out_dir}")
    return masks_generated

# ============================================================================
# MAIN FUNCTION - DO NOT MODIFY (Official Template)
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to test images folder")
    parser.add_argument("--masks", required=True, help="Path to masks folder (unused)")
    parser.add_argument("--out", required=True, help="Path to output predictions")
    args = parser.parse_args()

    # Build band → folder map
    imagepath = {}
    for band in os.listdir(args.data):
        band_path = os.path.join(args.data, band)
        if os.path.isdir(band_path):
            imagepath[band] = band_path

    print(f"Processing bands: {list(imagepath.keys())}")

    # Run mask generation and save predictions
    maskgeration(imagepath, args.out)


if __name__ == "__main__":
    main()
