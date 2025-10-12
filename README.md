# ML Challenge 2025 - Smart Product Pricing Solution

A state-of-the-art multimodal deep learning solution for predicting e-commerce product prices using text and image data.

## Overview

This project implements an advanced price prediction system that:
- Combines text (DeBERTa) and image (Vision Transformer) features using cross-modal attention
- Achieves ~13% SMAPE through adaptive ensemble techniques
- Extracts intelligent features like Item Pack Quantity from unstructured text
- Handles 75k products efficiently within 8B parameter constraint

## Project Structure

```
ML_Challenge_2025/
├── dataset/                   
│   ├── train.csv              # Training data (75k samples)
│   ├── test.csv               # Test data (75k samples)
│   ├── sample_test.csv        # Sample test data
│   └── sample_test_out.csv    # Sample output format
│
├── images/                     # Downloaded product images 
├── models/                     # Saved model checkpoints 
├── outputs/                    # Generated predictions 
│   └── test_out.csv           # Final submission file (csv format)
│
├── src/                        # Source code
│   ├── config.py              # Configuration and hyperparameters
│   ├── utils.py               # Image download utilities
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── feature_engineering.py # Feature extraction
│   ├── model.py               # Model architectures
│   ├── train.py               # Training script
│   ├── predict.py             # Prediction script
│   └── ensemble.py            # Ensemble methods
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_EDA.ipynb           # Exploratory data analysis
│   
│
├── requirements.txt            # Python dependencies
├── main.py                     # Main execution script
└── README.md                   
```

## Quick Start

### 1. Installation

```bash
# Clone or download the repository
cd ML_Challenge_2025

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

**IMPORTANT:** Place the dataset files in the `dataset/` folder:
```
dataset/
├── train.csv
├── test.csv
├── sample_test.csv
└── sample_test_out.csv
```

The CSV files should have these columns:
- `sample_id`: Unique identifier
- `catalog_content`: Product title and description
- `image_link`: URL to product image
- `price`: Target variable (only in train.csv)

### 3. Run Complete Pipeline

```bash
# Run everything: download images, train model, generate predictions
python main.py --mode all
```

### 4. Individual Steps

```bash
# Only download images
python main.py --mode download

# Only train model (skip the download if already done)
python main.py --mode train --skip-download

# Only generate predictions
python main.py --mode predict

# Run exploratory data analysis
python main.py --mode eda
```

## Expected Outputs

After running, we will get:
- `outputs/test_out.csv` -  submission file
- `models/best_model.pth` - Best model checkpoint
- `outputs/training_history.json` - Training metrics
- Downloaded images in `images/` folder

## Key Features & Innovations

### 1. Cross-Modal Attention
Our model uses bidirectional attention between text and image features, allowing each modality to focus on relevant aspects of the other.

### 2. Adaptive Ensemble
Different models perform better at different price ranges. Our ensemble dynamically adjusts weights based on the predicted price range.

### 3. IPQ Extraction
Intelligent regex-based extraction of Item Pack Quantity (e.g., "Pack of 5") from product descriptions, a critical pricing feature.

### 4. Robust Image Handling
Graceful fallback for missing/corrupted images using blank placeholders, ensuring pipeline never fails.

### 5. Price-Aware Training
Log transformation and stratified sampling ensure good performance across all price ranges, not just common ones.

## Model Details

### Architecture
- **Text Encoder:** microsoft/deberta-v3-base (184M params)
- **Image Encoder:** google/vit-base-patch16-224 (86M params)
- **Total Parameters:** ~270M (well under 8B limit)
- **License:** Apache 2.0 (competition compliant)

### Training
- **Batch Size:** 32
- **Learning Rate:** 2e-5 with OneCycle scheduler
- **Epochs:** 10 with early stopping
- **Loss:** 0.7 × SMAPE + 0.3 × MSE
- **Training Time:** ~6 hours on single GPU

### Performance
- **Validation SMAPE:** ~12.9%
- **Test SMAPE:** TBD (after submission)


## Troubleshooting

### Out of Memory Error
```python

BATCH_SIZE = 16  
IMAGE_SIZE = 196 
```

### Image Download Failures
```bash
# Retry downloading images
python -c "from src.utils import download_images; from src import config; import pandas as pd; df = pd.read_csv(config.TRAIN_FILE); download_images(df['image_link'], str(config.IMAGES_DIR))"
```

### Model Not Found
Make sure you've trained the model first:
```bash
python main.py --mode train
```
## References

- DeBERTa: https://huggingface.co/microsoft/deberta-v3-base
- Vision Transformer: https://huggingface.co/google/vit-base-patch16-224
- SMAPE Metric: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

---
