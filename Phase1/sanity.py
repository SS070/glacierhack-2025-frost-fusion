import torch

# Load checkpoint
checkpoint = torch.load('model.pth', map_location='cpu', weights_only=False)

print("✅ Checkpoint loaded successfully!")
print(f"\nCheckpoint keys: {checkpoint.keys()}")

if 'model_state_dict' in checkpoint:
    print(f"\n✅ Model state dict found")
    print(f"   - Parameters: {len(checkpoint['model_state_dict'])}")
    print(f"   - Threshold: {checkpoint.get('threshold', 'NOT FOUND')}")
    print(f"   - Best MCC: {checkpoint.get('best_mcc', 'NOT FOUND')}")
    print(f"   - Epoch: {checkpoint.get('epoch', 'NOT FOUND')}")

    # Check architecture compatibility
    expected_keys = ['inc.double_conv.0.weight', 'down1.maxpool_conv.1.double_conv.0.weight']
    found = all(key in checkpoint['model_state_dict'] for key in expected_keys)

    if found:
        print(f"\n✅ Architecture matches EfficientUNet")
    else:
        print(f"\n❌ Architecture mismatch!")
else:
    print(f"\n❌ 'model_state_dict' key not found!")

# Check file size
import os

file_size = os.path.getsize('model.pth') / (1024 * 1024)
print(f"\n📦 File size: {file_size:.2f} MB")

if file_size > 200:
    print("❌ WARNING: File exceeds 200 MB limit!")
elif file_size < 1:
    print("⚠️  WARNING: File seems too small!")
else:
    print("✅ File size is acceptable")
