"""
Test your solution.py locally before submission
This simulates the evaluation environment
"""

import os
import sys
import glob
import tifffile
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, jaccard_score

# Import your solution
import solution


def compute_metrics(pred_mask, true_mask):
    """Compute evaluation metrics"""
    pred_flat = (pred_mask > 127).astype(int).flatten()
    true_flat = (true_mask > 0).astype(int).flatten()

    if len(np.unique(pred_flat)) == 1 or len(np.unique(true_flat)) == 1:
        return {'mcc': 0.0, 'f1': 0.0, 'iou': 0.0}

    mcc = matthews_corrcoef(true_flat, pred_flat)
    f1 = f1_score(true_flat, pred_flat, zero_division=0)
    iou = jaccard_score(true_flat, pred_flat, zero_division=0)

    return {'mcc': mcc, 'f1': f1, 'iou': iou}


def test_on_train_data():
    """Test solution on training data"""
    print("=" * 70)
    print("TESTING SOLUTION ON TRAINING DATA")
    print("=" * 70)

    data_dir = r"C:\S\Programming\glacier_hack\Train"
    output_dir = r"C:\S\Programming\glacier_hack\test_output"

    os.makedirs(output_dir, exist_ok=True)

    # Prepare imagepath dictionary
    imagepath = {
        'Band1': os.path.join(data_dir, 'Band1'),
        'Band2': os.path.join(data_dir, 'Band2'),
        'Band3': os.path.join(data_dir, 'Band3'),
        'Band4': os.path.join(data_dir, 'Band4'),
        'Band5': os.path.join(data_dir, 'Band5')
    }

    print("\nRunning maskgeration...")
    solution.maskgeration(imagepath, output_dir)  # No return value expected

    # Load saved masks from output directory
    output_files = glob.glob(os.path.join(output_dir, '*.tif'))
    print(f"\nGenerated {len(output_files)} mask files")

    # Load ground truth and compare
    label_dir = os.path.join(data_dir, 'label')
    label_files = glob.glob(os.path.join(label_dir, '*.tif'))

    all_mccs = []
    all_f1s = []
    all_ious = []

    print("\nEvaluating predictions...")
    for label_file in label_files:
        filename = os.path.basename(label_file)

        # Extract tile ID
        import re
        match = re.search(r'(\d{2}_\d{2})', filename)
        if not match:
            continue
        tile_id = match.group(1)

        # Find corresponding prediction file
        pred_files = [f for f in output_files if tile_id in f]
        if not pred_files:
            print(f"Warning: No prediction for {tile_id}")
            continue

        pred_file = pred_files[0]

        # Load ground truth
        true_mask = tifffile.imread(label_file).astype(np.float32)
        if true_mask.ndim == 3:
            true_mask = true_mask[..., 0]

        # Load prediction
        pred_mask = tifffile.imread(pred_file).astype(np.float32)
        if pred_mask.ndim == 3:
            pred_mask = pred_mask[..., 0]

        # Compute metrics
        metrics = compute_metrics(pred_mask, true_mask)
        all_mccs.append(metrics['mcc'])
        all_f1s.append(metrics['f1'])
        all_ious.append(metrics['iou'])

        print(f"Tile {tile_id}: MCC={metrics['mcc']:.4f}, F1={metrics['f1']:.4f}, IoU={metrics['iou']:.4f}")

    # Overall metrics
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Mean MCC: {np.mean(all_mccs):.4f} +/- {np.std(all_mccs):.4f}")
    print(f"Mean F1:  {np.mean(all_f1s):.4f} +/- {np.std(all_f1s):.4f}")
    print(f"Mean IoU: {np.mean(all_ious):.4f} +/- {np.std(all_ious):.4f}")
    print("=" * 70)

def test_command_line_interface():
    """Test command-line interface"""
    print("\n" + "=" * 70)
    print("TESTING COMMAND-LINE INTERFACE")
    print("=" * 70)

    data_dir = r"C:\S\Programming\glacier_hack\Train"
    output_dir = r"C:\S\Programming\glacier_hack\cli_test_output"

    # Simulate command line call
    sys.argv = ['solution.py', '--imagepath', data_dir, '--output', output_dir]

    try:
        solution.main()
        print("Command-line interface test: PASSED")
    except Exception as e:
        print(f"Command-line interface test: FAILED")
        print(f"Error: {e}")


if __name__ == "__main__":
    # Test on training data
    test_on_train_data()

    # Test CLI
    test_command_line_interface()