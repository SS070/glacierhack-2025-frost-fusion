# ============================================================================
# GLACIER SEGMENTATION - IMPROVED PATCH-BASED CNN (TRAIN - V2)
# Advanced patch-based approach with all optimizations
# ============================================================================

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tifffile
from PIL import Image
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR SMALL DATASET
# ============================================================================

PATCH_SIZE = 48
INPUT_SIZE = 256
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 250
NUM_WORKERS = 0
PATCH_STRIDE = 12

# Class configuration
CLASS_VALUES = np.array([0, 85, 170, 255], dtype=np.float32)
CLASS_MAP = {0: 0, 85: 1, 170: 2, 255: 3}
CLASS_WEIGHTS = torch.tensor([0.5, 2.5, 5.0, 10.0], dtype=torch.float32)

# Learning rate
WARMUP_EPOCHS = 15
LEARNING_RATE_WARMUP = 0.00005
LEARNING_RATE_MAX = 0.0005
GRAD_CLIP_NORM = 1.0

# Loss configuration
FOCAL_LOSS = True
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 2.5

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ============================================================================
# IMPROVED PATCH CNN - DEEPER & BETTER REGULARIZED
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
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.dropout3(x)
        
        # Block 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.dropout4(x)
        
        # Global pool and FC
        x = self.global_pool(x).flatten(1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        
        return x

# ============================================================================
# DATASET WITH AGGRESSIVE PATCH EXTRACTION
# ============================================================================

class ImprovedGlacierDataset(Dataset):
    def __init__(self, img_ids, train_dir, patch_size=48, stride=12, augment=False):
        self.img_ids = img_ids
        self.train_dir = train_dir
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.bands = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5']
        self.label_dir = os.path.join(train_dir, 'labels')
        self.patches = []
        self.prepare_patches()
        
    def load_band(self, img_id, band_name):
        path = os.path.join(self.train_dir, band_name, f'img{img_id}.tif')
        try:
            img = tifffile.imread(path)
        except:
            img = np.array(Image.open(path))
        
        if img.ndim == 3:
            img = img[..., 0]
        
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / (img.max() + 1e-8)
        
        return img
    
    @staticmethod
    def label_patch_to_class(label_patch):
        vals, counts = np.unique(label_patch.reshape(-1), return_counts=True)
        if len(vals) == 0:
            return 0
        mode_val = vals[counts.argmax()].astype(np.float32)
        diffs = np.abs(CLASS_VALUES - mode_val)
        nearest_val = int(CLASS_VALUES[np.argmin(diffs)])
        return CLASS_MAP.get(nearest_val, 0)
    
    def prepare_patches(self):
        for img_id in self.img_ids:
            bands = [self.load_band(img_id, band) for band in self.bands]
            
            label_path = os.path.join(self.label_dir, f'img{img_id}.tif')
            try:
                label = tifffile.imread(label_path)
            except:
                label = np.array(Image.open(label_path))
            
            if label.ndim == 3:
                label = label[..., 0]
            
            # Dense patch extraction
            for x in range(0, INPUT_SIZE - self.patch_size + 1, self.stride):
                for y in range(0, INPUT_SIZE - self.patch_size + 1, self.stride):
                    patch_data = np.stack([b[x:x+self.patch_size, y:y+self.patch_size] for b in bands])
                    label_patch = label[x:x+self.patch_size, y:y+self.patch_size]
                    class_idx = self.label_patch_to_class(label_patch)
                    self.patches.append((patch_data, class_idx))
    
    def augment_patch(self, patch):
        # Rotation
        if np.random.rand() > 0.5:
            k = np.random.randint(0, 4)
            patch = torch.rot90(patch, k, dims=[1, 2])
        
        # Flips
        if np.random.rand() > 0.5:
            patch = torch.flip(patch, dims=[2])
        if np.random.rand() > 0.5:
            patch = torch.flip(patch, dims=[1])
        
        # Brightness
        if np.random.rand() > 0.5:
            factor = 0.8 + 0.4 * np.random.rand()
            patch = patch * factor
        
        # Contrast
        if np.random.rand() > 0.5:
            factor = 0.9 + 0.2 * np.random.rand()
            mean = patch.mean()
            patch = (patch - mean) * factor + mean
        
        # Noise
        if np.random.rand() > 0.6:
            patch = patch + torch.randn_like(patch) * 0.02
        
        return torch.clamp(patch, 0, 1)
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch, label = self.patches[idx]
        patch = torch.from_numpy(patch).float()
        if self.augment:
            patch = self.augment_patch(patch)
        return patch, torch.tensor(label, dtype=torch.long)

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def calculate_mcc(y_true, y_pred):
    try:
        return matthews_corrcoef(y_true, y_pred)
    except:
        return 0.0

def get_lr(epoch, total_epochs, warmup_epochs, lr_warmup, lr_max):
    if epoch < warmup_epochs:
        return lr_warmup + (lr_max - lr_warmup) * epoch / max(1, warmup_epochs)
    else:
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return lr_max * 0.5 * (1 + np.cos(np.pi * t))

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Train', ncols=100)
    for patches, labels in pbar:
        patches = patches.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(patches)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    mcc = calculate_mcc(all_labels, all_preds)
    
    return total_loss / max(1, len(loader)), accuracy, mcc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Val', ncols=100)
    for patches, labels in pbar:
        patches = patches.to(device)
        labels = labels.to(device)
        
        outputs = model(patches)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    mcc = calculate_mcc(all_labels, all_preds)
    
    return total_loss / max(1, len(loader)), accuracy, mcc

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='model.pth')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("GLACIER SEGMENTATION - IMPROVED PATCH-BASED CNN (V2)")
    print(f"{'='*80}")
    print(f"✓ Device: {DEVICE}")
    print(f"✓ Patch size: {PATCH_SIZE}×{PATCH_SIZE}")
    print(f"✓ Stride: {PATCH_STRIDE}")
    print(f"✓ Focal Loss: {FOCAL_LOSS}")
    print(f"✓ Advanced augmentation and dropout\n")
    
    # Get image IDs
    band1_path = os.path.join(args.train_data, 'Band1')
    img_ids = sorted([f.replace('.tif', '').replace('img', '')
                     for f in os.listdir(band1_path) if f.endswith('.tif')])
    
    print(f"✓ Found {len(img_ids)} images")
    
    # Train/val split
    train_size = int(0.8 * len(img_ids))
    train_ids = img_ids[:train_size]
    val_ids = img_ids[train_size:]
    
    print(f"✓ Train: {len(train_ids)}, Val: {len(val_ids)}\n")
    
    # Datasets
    print("Preparing training patches...")
    train_dataset = ImprovedGlacierDataset(train_ids, args.train_data, PATCH_SIZE, PATCH_STRIDE, augment=True)
    print(f"✓ {len(train_dataset)} training patches\n")
    
    print("Preparing validation patches...")
    val_dataset = ImprovedGlacierDataset(val_ids, args.train_data, PATCH_SIZE, PATCH_STRIDE, augment=False)
    print(f"✓ {len(val_dataset)} validation patches\n")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model
    model = ImprovedPatchCNN(in_channels=5, num_classes=NUM_CLASSES).to(DEVICE)
    
    # Criterion
    if FOCAL_LOSS:
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, weight=CLASS_WEIGHTS.to(DEVICE))
    else:
        criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_WARMUP, weight_decay=1e-5)
    
    print(f"{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")
    
    best_val_mcc = -1.0
    patience = 50
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        lr = get_lr(epoch - 1, EPOCHS, WARMUP_EPOCHS, LEARNING_RATE_WARMUP, LEARNING_RATE_MAX)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        train_loss, train_acc, train_mcc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_mcc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"\nEpoch {epoch}/{EPOCHS} | LR: {lr:.7f}")
        print(f" Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | MCC: {train_mcc:.4f}")
        print(f" Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | MCC: {val_mcc:.4f}")
        print(f" Gap   - MCC: {train_mcc - val_mcc:.4f}")
        
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            patience_counter = 0
            torch.save(model.state_dict(), args.model_path)
            print(f" ✓ Best model saved! (Val MCC: {best_val_mcc:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch}")
            break
    
    print(f"\n{'='*80}")
    print("✓ TRAINING COMPLETED")
    print(f"✓ Best Val MCC: {best_val_mcc:.4f}")
    print(f"✓ Model saved to: {args.model_path}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
