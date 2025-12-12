# ============================================================================
# GLACIER SEGMENTATION - PHASE 1 SOLUTION (SUBMISSION - V3)
# Direct 512×512 inference with maskgeneration function
# ============================================================================

import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from PIL import Image

PATCH_SIZE = 64
IMAGE_SIZE = 512
NUM_CLASSES = 4
CLASS_TO_PIXEL = {0: 0, 1: 85, 2: 170, 3: 255}
DEVICE = torch.device('cpu')

# ============================================================================
# MODEL (EXACT MATCH WITH TRAIN/TEST)
# ============================================================================

class OptimizedPatchCNN(nn.Module):
    def __init__(self, in_channels=5, num_classes=4):
        super(OptimizedPatchCNN, self).__init__()
        
        # Block 1
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.4)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.4)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.5)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout2d(0.5)
        
        # Global pool and FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.dropout4(x)
        
        x = self.global_pool(x).flatten(1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        
        return x

# ============================================================================
# UTILITIES
# ============================================================================

def load_band(path):
    """Load and normalize band image to 512×512"""
    try:
        img = tifffile.imread(path)
    except:
        img = np.array(Image.open(path))
    
    if img.ndim == 3:
        img = img[..., 0]
    
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / (img.max() + 1e-8)
    
    # Resize to 512×512
    if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
        img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
        )).astype(np.float32) / 255.0
    
    return img

# ============================================================================
# INFERENCE WITH 8-FOLD TTA
# ============================================================================

@torch.no_grad()
def predict_mask_tta(model, bands, device):
    """
    Predict mask with 8-fold TTA
    Args:
        model: Trained model
        bands: List of 5 numpy arrays (512×512 each, normalized)
        device: torch device
    Returns:
        mask: numpy array (512×512, dtype=uint8) with pixel values {0, 85, 170, 255}
    """
    
    mask_votes = np.zeros((IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES), dtype=np.float32)
    patch_size = PATCH_SIZE
    stride = 16
    
    augmentations = [
        (False, False, 0),  # Original
        (True, False, 0),   # H-flip
        (False, True, 0),   # V-flip
        (True, True, 0),    # H+V flip
        (False, False, 1),  # 90°
        (True, False, 1),   # 90° + H-flip
        (False, True, 1),   # 90° + V-flip
        (True, True, 1),    # 90° + H+V flip
    ]
    
    for flip_h, flip_v, rot_k in augmentations:
        # Apply flips
        aug_bands = [b.copy() for b in bands]
        if flip_h:
            aug_bands = [np.fliplr(b) for b in aug_bands]
        if flip_v:
            aug_bands = [np.flipud(b) for b in aug_bands]
        
        # Apply rotation
        if rot_k > 0:
            for _ in range(rot_k):
                aug_bands = [np.rot90(b) for b in aug_bands]
        
        # Sliding window prediction
        for x in range(0, IMAGE_SIZE - patch_size + 1, stride):
            for y in range(0, IMAGE_SIZE - patch_size + 1, stride):
                patch = np.stack([b[x:x+patch_size, y:y+patch_size] for b in aug_bands])
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
                
                output = model(patch_tensor)
                probs = F.softmax(output, dim=1)[0].cpu().numpy()
                
                # Reverse augmentations for probability map
                if rot_k > 0:
                    for _ in range(rot_k):
                        probs = np.rot90(probs, axes=(1, 2), k=3)
                if flip_h:
                    probs = np.fliplr(probs)
                if flip_v:
                    probs = np.flipud(probs)
                
                # Add to voting
                mask_votes[x:x+patch_size, y:y+patch_size, :] += probs
    
    # Average and get argmax
    mask_votes /= len(augmentations)
    mask_classes = np.argmax(mask_votes, axis=2)
    
    # Convert to pixel values
    mask_pixels = np.vectorize(CLASS_TO_PIXEL.get)(mask_classes).astype(np.uint8)
    
    return mask_pixels

# ============================================================================
# MAIN SUBMISSION FUNCTION (REQUIRED)
# ============================================================================

def maskgeneration(imagepath, model_path):
    """
    Main function for mask generation (REQUIRED SUBMISSION FORMAT)
    
    Args:
        imagepath: Dictionary with band folder paths {Band1, Band2, ...}
        model_path: Path to trained model.pth
    
    Returns:
        Dictionary mapping image IDs to 512×512 masks with values {0, 85, 170, 255}
    """
    
    # Load model
    model = OptimizedPatchCNN(in_channels=5, num_classes=NUM_CLASSES).to(DEVICE)
    
    state_dict = torch.load(model_path, map_location=DEVICE)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Get band folder paths
    bands = sorted([b for b in imagepath.keys() if 'Band' in b],
                  key=lambda x: int(re.search(r'\d+', x).group()))
    
    if not bands or len(bands) < 5:
        raise ValueError("Expected at least 5 band folders")
    
    # Get image filenames from Band1
    band1_dir = imagepath[bands[0]]
    filenames = sorted([f for f in os.listdir(band1_dir) if f.endswith('.tif')])
    
    masks = {}
    
    for filename in filenames:
        try:
            # Load all 5 bands (enforced 512×512)
            band_data = []
            valid = True
            
            for band_name in bands[:5]:
                band_path = os.path.join(imagepath[band_name], filename)
                
                if not os.path.exists(band_path):
                    valid = False
                    break
                
                try:
                    band_array = load_band(band_path)
                    band_data.append(band_array)
                except Exception as e:
                    valid = False
                    break
            
            if not valid or len(band_data) != 5:
                # Return default mask if loading fails
                tile_id = filename.replace('img', '').replace('.tif', '')
                masks[tile_id] = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                continue
            
            # Predict mask with 8-fold TTA (returns 512×512)
            mask_pixels = predict_mask_tta(model, band_data, DEVICE)
            
            # Extract tile ID and store
            tile_id = filename.replace('img', '').replace('.tif', '')
            masks[tile_id] = mask_pixels
            
        except Exception as e:
            # Return default mask on any error
            tile_id = filename.replace('img', '').replace('.tif', '')
            masks[tile_id] = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            continue
    
    return masks

# ============================================================================
# HARNESS (DO NOT MODIFY)
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagepath', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    
    imagepath = {}
    for band in os.listdir(args.imagepath):
        band_path = os.path.join(args.imagepath, band)
        if os.path.isdir(band_path):
            imagepath[band] = band_path
    
    masks = maskgeneration(imagepath, args.model_path)
    
    print(f"Generated masks for {len(masks)} images")
    print("Masks dictionary returned successfully")
