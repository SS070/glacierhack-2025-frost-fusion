import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from PIL import Image

PATCH_SIZE = 48
INPUT_SIZE = 256
OUTPUT_SIZE = 512
NUM_CLASSES = 4
CLASS_TO_PIXEL = {0: 0, 1: 85, 2: 170, 3: 255}
DEVICE = torch.device('cpu')

# ============================================================================
# MODEL (EXACT MATCH WITH TRAIN/TEST)
# ============================================================================

class ImprovedPatchCNN(nn.Module):
    def __init__(self, in_channels=5, num_classes=4):
        super(ImprovedPatchCNN, self).__init__()
        
        # Block 1
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.3)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout2d(0.3)
        
        # Global pool and FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.4)
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
    """Load and normalize band image to 256×256"""
    try:
        img = tifffile.imread(path)
    except:
        img = np.array(Image.open(path))
    
    if img.ndim == 3:
        img = img[..., 0]
    
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / (img.max() + 1e-8)
    
    # Resize to 256×256 for model input
    if img.shape != (INPUT_SIZE, INPUT_SIZE):
        img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(
            (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR
        )).astype(np.float32) / 255.0
    
    return img

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

@torch.no_grad()
def predict_mask(model, bands_5, device):
    """
    Args:
        bands_5: list of 5 numpy arrays (256x256 each, normalized)
    Returns:
        mask: numpy array (512x512, dtype=uint8) with pixel values {0, 85, 170, 255}
    """
    
    img = np.stack(bands_5, axis=0)
    mask_classes = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    
    patches, positions = [], []
    
    # Sliding window prediction (no overlap for simplicity)
    for i in range(0, INPUT_SIZE, PATCH_SIZE):
        for j in range(0, INPUT_SIZE, PATCH_SIZE):
            i_end = min(i + PATCH_SIZE, INPUT_SIZE)
            j_end = min(j + PATCH_SIZE, INPUT_SIZE)
            
            patch = img[:, i:i_end, j:j_end]
            
            # Pad if necessary
            if patch.shape[1] < PATCH_SIZE or patch.shape[2] < PATCH_SIZE:
                padded = np.zeros((5, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                padded[:, :patch.shape[1], :patch.shape[2]] = patch
                patch = padded
            
            patches.append(patch)
            positions.append((i, j, i_end, j_end))
    
    # Batch prediction
    batch_tensor = torch.from_numpy(np.stack(patches)).float().to(device)
    outputs = model(batch_tensor)
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    # Fill mask
    for pred, (i, j, i_end, j_end) in zip(predictions, positions):
        mask_classes[i:i_end, j:j_end] = pred
    
    # Convert to pixel values
    mask_pixels = np.vectorize(CLASS_TO_PIXEL.get)(mask_classes).astype(np.uint8)
    
    # Resize from 256×256 to 512×512 using nearest neighbor
    mask_512 = np.array(Image.fromarray(mask_pixels).resize(
        (OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST
    ))
    
    return mask_512

# ============================================================================
# MAIN SUBMISSION FUNCTION
# ============================================================================

def maskgeration(imagepath, model_path):
    """
    Main function for mask generation (REQUIRED SUBMISSION FORMAT)
    
    Args:
        imagepath: Dictionary with band folder paths {Band1, Band2, ...}
        model_path: Path to trained model.pth
    
    Returns:
        Dictionary mapping image IDs to 512×512 masks with values {0, 85, 170, 255}
    """
    
    # Load model
    model = ImprovedPatchCNN(in_channels=5, num_classes=NUM_CLASSES).to(DEVICE)
    
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
            # Load all 5 bands (enforced 256×256)
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
                masks[tile_id] = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
                continue
            
            # Predict mask (returns 512×512)
            mask_pixels = predict_mask(model, band_data, DEVICE)
            
            # Extract tile ID and store
            tile_id = filename.replace('img', '').replace('.tif', '')
            masks[tile_id] = mask_pixels
            
        except Exception as e:
            # Return default mask on any error
            tile_id = filename.replace('img', '').replace('.tif', '')
            masks[tile_id] = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
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
