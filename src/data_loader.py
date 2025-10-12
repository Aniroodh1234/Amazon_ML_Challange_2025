import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import config

class ProductDataset(Dataset):
    """Custom Dataset for product pricing"""
    
    def __init__(self, df, text_tokenizer, image_transform, is_test=False):
        """
        Args:
            df: DataFrame with sample_id, catalog_content, image_link, and optionally price
            text_tokenizer: Tokenizer for text processing
            image_transform: Transform for image preprocessing
            is_test: Whether this is test data (no prices)
        """
        self.df = df.reset_index(drop=True)
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def extract_ipq(self, catalog_content):
        """Extract Item Pack Quantity from catalog content"""
        # Look for patterns like "Pack of 5", "5 Pack", "Quantity: 5", etc.
        patterns = [
            r'pack\s+of\s+(\d+)',
            r'(\d+)\s+pack',
            r'quantity[:\s]+(\d+)',
            r'count[:\s]+(\d+)',
            r'ipq[:\s]+(\d+)',
        ]
        
        catalog_lower = catalog_content.lower()
        for pattern in patterns:
            match = re.search(pattern, catalog_lower)
            if match:
                try:
                    return int(match.group(1))
                except:
                    pass
        return 1  # Default to 1 if no IPQ found
    
    def load_image(self, image_link):
        """Load image from local storage or return placeholder"""
        try:
            filename = Path(image_link).name
            image_path = config.IMAGES_DIR / filename
            
            if image_path.exists():
                image = Image.open(image_path).convert('RGB')
                return image
            else:
                # Return a blank placeholder image
                return Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='white')
        except Exception as e:
            # Return a blank placeholder on error
            return Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), color='white')
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text processing
        catalog_content = str(row['catalog_content'])
        text_encoding = self.text_tokenizer(
            catalog_content,
            max_length=config.MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Image processing
        image = self.load_image(row['image_link'])
        image_tensor = self.image_transform(image)
        
        # Extract IPQ feature
        ipq = self.extract_ipq(catalog_content)
        
        # Prepare output
        item = {
            'sample_id': row['sample_id'],
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'image': image_tensor,
            'ipq': torch.tensor(ipq, dtype=torch.float32),
        }
        
        if not self.is_test:
            # Log transform price for better distribution
            price = row['price']
            if config.PRICE_LOG_TRANSFORM:
                price = np.log1p(price)
            item['price'] = torch.tensor(price, dtype=torch.float32)
        
        return item


def load_data():
    """Load and preprocess all datasets"""
    print("Loading datasets...")
    
    # Load training data
    train_df = pd.read_csv(config.TRAIN_FILE)
    print(f"Training data shape: {train_df.shape}")
    
    # Load test data
    test_df = pd.read_csv(config.TEST_FILE)
    print(f"Test data shape: {test_df.shape}")
    
    # Basic data cleaning
    train_df = train_df.dropna(subset=['catalog_content', 'image_link', 'price'])
    train_df = train_df[train_df['price'] > 0]  # Remove invalid prices
    
    # Remove outliers (prices beyond 3 standard deviations)
    price_mean = train_df['price'].mean()
    price_std = train_df['price'].std()
    train_df = train_df[
        (train_df['price'] >= price_mean - 3*price_std) & 
        (train_df['price'] <= price_mean + 3*price_std)
    ]
    
    print(f"After cleaning, training data shape: {train_df.shape}")
    print(f"Price statistics:")
    print(train_df['price'].describe())
    
    # Train-validation split
    train_data, val_data = train_test_split(
        train_df, 
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_SEED
    )
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    
    return train_data, val_data, test_df


def get_text_length_stats(df):
    """Analyze text length distribution"""
    lengths = df['catalog_content'].apply(lambda x: len(str(x).split()))
    return {
        'mean': lengths.mean(),
        'median': lengths.median(),
        'max': lengths.max(),
        'min': lengths.min()
    }


def analyze_price_distribution(df):
    """Analyze price distribution for better modeling"""
    prices = df['price']
    
    analysis = {
        'mean': prices.mean(),
        'median': prices.median(),
        'std': prices.std(),
        'min': prices.min(),
        'max': prices.max(),
        'q25': prices.quantile(0.25),
        'q75': prices.quantile(0.75),
        'skewness': prices.skew(),
    }
    
    print("\nPrice Distribution Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value:.2f}")
    
    return analysis


if __name__ == "__main__":
    # Test data loading
    train_data, val_data, test_data = load_data()
    analyze_price_distribution(train_data)
    text_stats = get_text_length_stats(train_data)
    print(f"\nText Length Statistics: {text_stats}")