# Easy Bank Statement Parser

A YOLO-based table detection system for bank statement parsing using computer vision.

## Features

- **Table Detection**: Uses YOLOv8 to detect table regions in bank statement images
- **High Accuracy**: Achieves 99.5% mAP50 on validation and test sets
- **Multi-Bank Support**: Works with various bank statement formats (CBI, Canara Bank, Indian Bank, IndusInd Bank, Jammu Kashmir Bank)
- **Fast Inference**: ~118ms per image on CPU

## Performance Metrics

- **Precision**: 99.8%
- **Recall**: 100%
- **mAP50**: 99.5%
- **mAP50-95**: 99.5%

## Project Structure

```
bankStatementParser/
├── datasets/
│   └── bank_statements/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
├── runs/detect/train3/weights/best.pt  # Trained model
├── train_yolo.py                       # Training script
├── evaluate_model.py                   # Evaluation script
├── bankStatements.yaml                 # Dataset configuration
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install ultralytics
```

### 2. Train Model

```bash
python train_yolo.py
```

### 3. Evaluate Model

```bash
python evaluate_model.py
```

### 4. Use Trained Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train3/weights/best.pt')

# Predict on new images
results = model.predict('path/to/bank_statement.jpg')
```

## Dataset

The dataset contains bank statement images from multiple Indian banks:
- **Training**: 245 images
- **Validation**: 31 images  
- **Test**: 31 images

Each image is annotated with bounding boxes around table regions.

## Model Details

- **Architecture**: YOLOv8n (nano)
- **Input Size**: 640x640
- **Training Epochs**: 10
- **Batch Size**: 8
- **Device**: CPU compatible

## Results

The model successfully detects table regions in bank statements with:
- Perfect recall (100%) - no missed tables
- High precision (99.8%) - minimal false positives
- Consistent performance across different bank formats

## Author

Affan Ahmed
