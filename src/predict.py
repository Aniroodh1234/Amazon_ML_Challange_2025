import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ViTImageProcessor
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

import config
from data_loader import ProductDataset
from model import get_model


def load_trained_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    model = get_model('multimodal').to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'smape' in checkpoint:
        print(f"Model validation SMAPE: {checkpoint['smape']:.2f}%")
    
    return model


def predict(model, test_loader, device):
    """Generate predictions on test set"""
    model.eval()
    all_predictions = []
    all_sample_ids = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image = batch['image'].to(device)
            ipq = batch['ipq'].to(device)
            sample_ids = batch['sample_id']
            
            # Forward pass
            price_pred, _ = model(input_ids, attention_mask, image, ipq)
            
            # Store predictions
            predictions = price_pred.squeeze().cpu().numpy()
            
            # Handle single sample case
            if predictions.shape == ():
                predictions = [predictions.item()]
            
            all_predictions.extend(predictions)
            all_sample_ids.extend(sample_ids)
    
    # Convert to numpy array
    all_predictions = np.array(all_predictions)
    
    # Inverse log transform if needed
    if config.PRICE_LOG_TRANSFORM:
        all_predictions = np.expm1(all_predictions)
    
    # Ensure all predictions are positive
    all_predictions = np.maximum(all_predictions, 0.01)
    
    return all_sample_ids, all_predictions


def create_submission(sample_ids, predictions, output_path):
    """Create submission CSV file"""
    # Create DataFrame
    submission_df = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
    # Round prices to specified decimal places
    submission_df['price'] = submission_df['price'].round(config.SUBMISSION_DECIMAL_PLACES)
    
    # Sort by sample_id to match test.csv order
    submission_df = submission_df.sort_values('sample_id').reset_index(drop=True)
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"\nSubmission file created: {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nPrediction statistics:")
    print(submission_df['price'].describe())
    print(f"\nSample predictions:")
    print(submission_df.head(10))
    
    return submission_df


def main():
    """Main prediction function"""
    print("="*50)
    print("Generating Predictions for Test Set")
    print("="*50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(config.TEST_FILE)
    print(f"Test data shape: {test_df.shape}")
    
    # Initialize tokenizers and processors
    print("\nInitializing tokenizers and processors...")
    text_tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    image_processor = ViTImageProcessor.from_pretrained(config.IMAGE_MODEL_NAME)
    
    # Image transforms
    image_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
    ])
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = ProductDataset(test_df, text_tokenizer, image_transform, is_test=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Load trained model
    best_model_path = config.MODELS_DIR / 'best_model.pth'
    
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {best_model_path}. "
            "Please train the model first using train.py"
        )
    
    model = load_trained_model(best_model_path, device)
    
    # Generate predictions
    sample_ids, predictions = predict(model, test_loader, device)
    
    # Create submission file
    submission_df = create_submission(sample_ids, predictions, config.OUTPUT_FILE)
    
    # Verify submission format
    print("\n" + "="*50)
    print("Submission Verification")
    print("="*50)
    
    # Check if all test samples are included
    if len(submission_df) == len(test_df):
        print("✓ All test samples included")
    else:
        print(f"✗ Warning: Expected {len(test_df)} samples, got {len(submission_df)}")
    
    # Check for missing values
    if submission_df['price'].isna().sum() == 0:
        print("✓ No missing predictions")
    else:
        print(f"✗ Warning: {submission_df['price'].isna().sum()} missing predictions")
    
    # Check for negative prices
    if (submission_df['price'] > 0).all():
        print("✓ All prices are positive")
    else:
        print(f"✗ Warning: {(submission_df['price'] <= 0).sum()} non-positive prices")
    
    print("\n✓ Submission file ready for upload!")
    print(f"File location: {config.OUTPUT_FILE}")


if __name__ == "__main__":
    main()