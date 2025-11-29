from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='bankStatements.yaml',
    epochs=10,
    imgsz=640,
    batch=8,
    device='cpu'  # Change to 'cuda' if GPU available
)

# Validate the model
metrics = model.val()

# Test on test set
test_results = model.predict('datasets/bank_statements/images/test', save=True)

print("Training completed!")
print(f"Best mAP50: {metrics.box.map50}")
print(f"Best mAP50-95: {metrics.box.map}")