import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
import glob
import tifffile
import cv2
from sklearn.metrics import matthews_corrcoef, f1_score, jaccard_score
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = r"C:\S\Programming\glacier_hack\Train"
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 3000
INITIAL_LR = 3e-4
WEIGHT_DECAY = 1e-4
N_FOLDS = 5
MODEL_PATH = 'model.pth'


# ============================================================================
# MODEL ARCHITECTURE - FIXED
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
        # FIXED: Changed from in_channels to in_channels // 2
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
    def __init__(self, n_channels=5, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

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
# DATASET
# ============================================================================

def robust_normalize(img):
    """Robust normalization"""
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


class GlacierDataset(Dataset):
    def __init__(self, data_dir, transform=None, img_size=256):
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size

        # Get all tile IDs from Band1
        band1_files = sorted(glob.glob(os.path.join(data_dir, 'Band1', '*.tif')))
        self.tile_ids = []

        for f in band1_files:
            filename = os.path.basename(f)
            import re
            match = re.search(r'(\d{2}_\d{2})', filename)
            if match:
                self.tile_ids.append(match.group(1))

        print(f"Found {len(self.tile_ids)} tiles")

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]

        # Load all 5 bands
        bands = []
        for band_num in [2, 3, 4, 6, 10]:
            if band_num == 2:
                band_folder = 'Band1'
                prefix = 'B2_B2_masked_'
            elif band_num == 3:
                band_folder = 'Band2'
                prefix = 'B3_B3_masked_'
            elif band_num == 4:
                band_folder = 'Band3'
                prefix = 'B4_B4_masked_'
            elif band_num == 6:
                band_folder = 'Band4'
                prefix = 'B6_B6_masked_'
            else:  # 10
                band_folder = 'Band5'
                prefix = 'B10_B10_masked_'

            band_path = os.path.join(self.data_dir, band_folder, f'{prefix}{tile_id}.tif')
            band_img = tifffile.imread(band_path).astype(np.float32)

            if band_img.ndim == 3:
                band_img = band_img[..., 0]

            band_img = robust_normalize(band_img)
            bands.append(band_img)

        # Load mask
        mask_path = os.path.join(self.data_dir, 'label', f'Y_output_resized_{tile_id}.tif')
        mask = tifffile.imread(mask_path).astype(np.float32)

        if mask.ndim == 3:
            mask = mask[..., 0]

        # Normalize mask to [0, 1]
        mask = (mask > 0).astype(np.float32)

        # Stack bands
        image = np.stack(bands, axis=-1)

        # Resize if needed
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # Add channel dimension to mask after augmentation
            mask = mask.unsqueeze(0)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return {'image': image, 'mask': mask, 'tile_id': tile_id}


def get_train_transforms(img_size):
    """Strong augmentations for small dataset"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
            A.GridDistortion(p=1),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ToTensorV2()
    ])


def get_val_transforms(img_size):
    """No augmentation for validation"""
    return A.Compose([
        ToTensorV2()
    ])


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return 0.5 * self.bce(pred, target) + 0.5 * self.dice(pred, target)


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(preds, targets, threshold=0.5):
    """Compute MCC, F1, IoU"""
    preds_sigmoid = torch.sigmoid(preds)
    preds_np = (preds_sigmoid > threshold).cpu().numpy().astype(int).flatten()
    targets_np = targets.cpu().numpy().astype(int).flatten()

    if len(np.unique(preds_np)) == 1 or len(np.unique(targets_np)) == 1:
        return {'mcc': 0.0, 'f1': 0.0, 'iou': 0.0}

    mcc = matthews_corrcoef(targets_np, preds_np)
    f1 = f1_score(targets_np, preds_np, zero_division=0)
    iou = jaccard_score(targets_np, preds_np, zero_division=0)

    return {'mcc': mcc, 'f1': f1, 'iou': iou}


def find_optimal_threshold(preds, targets):
    """Find threshold that maximizes MCC"""
    preds_sigmoid = torch.sigmoid(preds).cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()

    best_threshold = 0.5
    best_mcc = -1

    for threshold in np.arange(0.3, 0.8, 0.05):
        preds_binary = (preds_sigmoid > threshold).astype(int)
        if len(np.unique(preds_binary)) > 1 and len(np.unique(targets_np)) > 1:
            mcc = matthews_corrcoef(targets_np, preds_binary)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold

    return best_threshold, best_mcc


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_fold(train_loader, val_loader, fold_id=0):
    """Train single fold"""
    print(f"\n{'=' * 70}")
    print(f"FOLD {fold_id + 1}")
    print(f"{'=' * 70}")

    model = EfficientUNet(n_channels=5, n_classes=1).to(device)
    criterion = CombinedLoss()

    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    best_val_mcc = -1.0
    best_threshold = 0.5
    patience_counter = 0
    patience = 750

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch < 10:
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    all_preds.append(outputs)
                    all_targets.append(masks)

            val_loss /= len(val_loader)
            preds = torch.cat(all_preds, dim=0)
            targets = torch.cat(all_targets, dim=0)

            optimal_threshold, _ = find_optimal_threshold(preds, targets)
            metrics = compute_metrics(preds, targets, threshold=optimal_threshold)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:03d}/{EPOCHS} | "
                  f"LR: {current_lr:.2e} | "
                  f"TrLoss: {train_loss:.4f} | "
                  f"VLoss: {val_loss:.4f} | "
                  f"MCC: {metrics['mcc']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"IoU: {metrics['iou']:.4f} | "
                  f"Thr: {optimal_threshold:.2f}")

            if metrics['mcc'] > best_val_mcc:
                best_val_mcc = metrics['mcc']
                best_threshold = optimal_threshold
                patience_counter = 0

                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'best_mcc': best_val_mcc,
                    'threshold': best_threshold
                }
                torch.save(checkpoint, f'best_fold{fold_id}.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print(f"\nFold {fold_id + 1} Complete - Best MCC: {best_val_mcc:.4f} | Threshold: {best_threshold:.2f}")
    return best_val_mcc, best_threshold


# ============================================================================
# MAIN TRAINING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GLACIER SEGMENTATION TRAINING - OPTIMIZED FOR SMALL DATASET")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset: {DATA_DIR}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"K-Folds: {N_FOLDS}")
    print("=" * 70)

    # Load dataset
    train_transform = get_train_transforms(IMG_SIZE)
    val_transform = get_val_transforms(IMG_SIZE)

    full_dataset = GlacierDataset(DATA_DIR, transform=None, img_size=IMG_SIZE)
    total_size = len(full_dataset)

    print(f"\nTotal samples: {total_size}")

    # K-Fold Cross Validation
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    fold_thresholds = []
    fold_models = []

    start_time = time.time()

    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(total_size))):
        print(f"\nTrain samples: {len(train_ids)} | Val samples: {len(val_ids)}")

        # Create datasets with augmentation
        train_dataset = GlacierDataset(DATA_DIR, transform=train_transform, img_size=IMG_SIZE)
        val_dataset = GlacierDataset(DATA_DIR, transform=val_transform, img_size=IMG_SIZE)

        train_subset = torch.utils.data.Subset(train_dataset, train_ids)
        val_subset = torch.utils.data.Subset(val_dataset, val_ids)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=True)

        # Train fold
        mcc, threshold = train_fold(train_loader, val_loader, fold_id=fold)

        fold_results.append(mcc)
        fold_thresholds.append(threshold)

        # Load best model
        model = EfficientUNet(n_channels=5, n_classes=1).to(device)
        checkpoint = torch.load(f'best_fold{fold}.pth', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        fold_models.append(model)

    # Results summary
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")

    for i, (mcc, thr) in enumerate(zip(fold_results, fold_thresholds)):
        print(f"Fold {i + 1}: MCC={mcc:.4f}, Threshold={thr:.2f}")

    mean_mcc = np.mean(fold_results)
    std_mcc = np.std(fold_results)
    print(f"\nMean MCC: {mean_mcc:.4f} +/- {std_mcc:.4f}")

    # Save best single model
    best_fold = np.argmax(fold_results)
    print(f"\nBest single model: Fold {best_fold + 1} (MCC: {fold_results[best_fold]:.4f})")

    checkpoint = torch.load(f'best_fold{best_fold}.pth', map_location=device, weights_only=False)
    torch.save(checkpoint, MODEL_PATH)
    print(f"Saved to: {MODEL_PATH}")

    # Create ensemble model (average weights)
    ensemble_state = {}
    for key in fold_models[0].state_dict().keys():
        ensemble_state[key] = torch.stack([m.state_dict()[key].float() for m in fold_models]).mean(0)

    ensemble_checkpoint = {
        'model_state_dict': ensemble_state,
        'threshold': np.mean(fold_thresholds),
        'best_mcc': mean_mcc
    }
    torch.save(ensemble_checkpoint, 'model_ensemble.pth')
    print(f"Saved ensemble to: model_ensemble.pth")

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print(f"\nTotal time: {hours}h {minutes}m")
    print("=" * 70)