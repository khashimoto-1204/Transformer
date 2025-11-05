# Transformer-based RGB Image Classifier

This repository contains a training script that fine-tunes a Vision Transformer
(ViT-B/16) model on a custom RGB image dataset for multi-class classification.

## Installation

Create a Python environment (Python 3.9+ recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

Prepare your dataset with separate `train/` and `val/` splits and a directory
per class:

```
path/to/dataset/
├── train/
│   ├── class_a/
│   ├── class_b/
│   └── ...
└── val/
    ├── class_a/
    ├── class_b/
    └── ...
```

## Training

Run the training script by pointing it to your dataset directory and specifying
the number of classes:

```bash
python train_transformer_classifier.py path/to/dataset NUM_CLASSES \
    --output-dir outputs \
    --batch-size 32 \
    --epochs 20
```

Checkpoints and metrics are saved in the specified `output-dir`.
