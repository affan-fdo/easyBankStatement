from ultralytics import YOLO
import os

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Evaluate on validation set
print("Evaluating on validation set...")
val_metrics = model.val(data='bankStatements.yaml', split='val')

# Evaluate on test set
print("Evaluating on test set...")
test_metrics = model.val(data='bankStatements.yaml', split='test')

# Run inference on test images
print("Running inference on test images...")
test_results = model.predict('datasets/bank_statements/images/test', save=True, conf=0.5)

print(f"Validation mAP50: {val_metrics.box.map50:.4f}")
print(f"Test mAP50: {test_metrics.box.map50:.4f}")
print("Results saved in runs/detect/predict/")