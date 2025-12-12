# ============================================================================
# GLACIER SEGMENTATION - PHASE 1 TEST (8-FOLD TTA)
# Direct 512×512 inference with 8-fold test-time augmentation
# ============================================================================

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import tifffile
from PIL import Image
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

PATCH_SIZE = 64
IMAGE_SIZE = 512
NUM_CLASSES = 4
CLASS_MAP = {0: 0, 85: 1, 170: 2, 255: 3}
CLASS_TO_PIXEL = {0: 0, 1: 85, 2: 170, 3: 255}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# MODEL (EXACT MATCH WITH TRAIN)
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

def calculate_mcc(y_true, y_pred):
    """Calculate Matthews Correlation Coefficient"""
    try:
        return matthews_corrcoef(y_true, y_pred)
    except:
        return 0.0

# ============================================================================
# INFERENCE WITH 8-FOLD TTA
# ============================================================================

@torch.no_grad()
def predict_image_tta(model, bands, device, patch_size=64, stride=16):
    """
    8-Fold TTA: 4 rotations × 2 flips
    1. Original
    2. H-flip
    3. V-flip
    4. H+V flip
    5. 90° + original
    6. 90° + H-flip
    7. 90° + V-flip
    8. 90° + H+V flip
    """
    
    mask_votes = np.zeros((IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES), dtype=np.float32)
    
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
    
    return mask_classes

# ============================================================================
# MAIN TESTING
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--label_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./predictions')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("GLACIER SEGMENTATION - PHASE 1 TEST (8-FOLD TTA)")
    print(f"{'='*80}")
    print(f"✓ Device: {DEVICE}")
    print(f"✓ Image size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"✓ Patch size: {PATCH_SIZE}×{PATCH_SIZE}")
    print(f"✓ Test-Time Augmentation: ENABLED (8-fold)")
    print(f"✓ Stride: 16\n")
    
    # Load model
    model = OptimizedPatchCNN(in_channels=5, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()
    
    print("✓ Model loaded\n")
    
    # Get image IDs
    band1_path = os.path.join(args.test_data, 'Band1')
    img_ids = sorted([f.replace('.tif', '').replace('img', '')
                     for f in os.listdir(band1_path) if f.endswith('.tif')])
    
    print(f"✓ Found {len(img_ids)} images\n")
    
    band_names = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5']
    image_metrics = []
    
    print(f"{'='*80}")
    print("TESTING PREDICTIONS WITH 8-FOLD TTA")
    print(f"{'='*80}\n")
    
    for img_id in tqdm(img_ids, desc='Testing', ncols=100):
        try:
            # Load bands
            bands = []
            for band_name in band_names:
                band_path = os.path.join(args.test_data, band_name, f'img{img_id}.tif')
                band = load_band(band_path)
                bands.append(band)
            
            # Predict with 8-fold TTA
            mask_classes = predict_image_tta(model, bands, DEVICE, PATCH_SIZE, stride=16)
            
            # Convert to pixel values
            mask_pixels = np.vectorize(CLASS_TO_PIXEL.get)(mask_classes).astype(np.uint8)
            
            # Save prediction
            output_path = os.path.join(args.output_dir, f'img{img_id}.tif')
            tifffile.imwrite(output_path, mask_pixels)
            
            # Evaluate if labels available
            if args.label_dir:
                label_path = os.path.join(args.label_dir, f'img{img_id}.tif')
                if os.path.exists(label_path):
                    try:
                        gt_label = load_band(label_path)
                        gt_classes = np.zeros_like(gt_label, dtype=np.int32)
                        for pixel_val, class_idx in CLASS_MAP.items():
                            gt_classes[np.isclose(gt_label, pixel_val)] = class_idx
                        
                        acc = (mask_classes.flatten() == gt_classes.flatten()).mean()
                        mcc = calculate_mcc(gt_classes.flatten(), mask_classes.flatten())
                        
                        image_metrics.append({'img_id': img_id, 'accuracy': acc, 'mcc': mcc})
                    except:
                        pass
        except Exception as e:
            continue
    
    print(f"\n✓ Predictions saved to: {args.output_dir}\n")
    
    # Print results
    if image_metrics:
        print(f"{'='*80}")
        print("EVALUATION RESULTS (WITH MCC)")
        print(f"{'='*80}\n")
        
        print(f"{'Image ID':<15} {'Accuracy':<15} {'MCC':<15}")
        print(f"{'-'*45}")
        for metric in image_metrics:
            print(f"{metric['img_id']:<15} {metric['accuracy']:.4f} {metric['mcc']:.4f}")
        
        avg_accuracy = np.mean([m['accuracy'] for m in image_metrics])
        avg_mcc = np.mean([m['mcc'] for m in image_metrics])
        
        print(f"{'-'*45}")
        print(f"{'Average':<15} {avg_accuracy:.4f} {avg_mcc:.4f} ⭐\n")
        
        print(f"{'='*80}")
        print(f"Average MCC: {avg_mcc:.4f} ⭐")
        print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
