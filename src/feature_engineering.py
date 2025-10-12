import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import config

class TextFeatureExtractor:
    """Extract features from catalog content"""
    
    def __init__(self):
        self.brand_patterns = [
            r'^([A-Z][A-Za-z]+)',  # Capitalized words at start
            r'by\s+([A-Z][A-Za-z]+)',  # "by BrandName"
        ]
    
    def extract_brand(self, text):
        """Try to extract brand name from text"""
        for pattern in self.brand_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return "Unknown"
    
    def extract_numeric_features(self, text):
        """Extract numeric features from text"""
        # Find all numbers in text
        numbers = re.findall(r'\d+\.?\d*', text)
        numbers = [float(n) for n in numbers if float(n) > 0]
        
        if numbers:
            return {
                'num_count': len(numbers),
                'max_number': max(numbers),
                'min_number': min(numbers),
                'avg_number': np.mean(numbers),
            }
        else:
            return {
                'num_count': 0,
                'max_number': 0,
                'min_number': 0,
                'avg_number': 0,
            }
    
    def extract_text_stats(self, text):
        """Extract basic text statistics"""
        words = str(text).split()
        return {
            'word_count': len(words),
            'char_count': len(text),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'capital_count': sum(1 for c in text if c.isupper()),
            'digit_count': sum(1 for c in text if c.isdigit()),
        }
    
    def extract_category_keywords(self, text):
        """Extract category-related keywords"""
        text_lower = text.lower()
        
        keywords = {
            'is_electronic': any(word in text_lower for word in ['laptop', 'phone', 'tablet', 'computer', 'electronic']),
            'is_clothing': any(word in text_lower for word in ['shirt', 'pant', 'dress', 'shoe', 'clothing']),
            'is_book': any(word in text_lower for word in ['book', 'novel', 'paperback', 'hardcover']),
            'is_food': any(word in text_lower for word in ['food', 'snack', 'drink', 'grocery']),
            'is_beauty': any(word in text_lower for word in ['beauty', 'cosmetic', 'makeup', 'skincare']),
            'is_home': any(word in text_lower for word in ['furniture', 'home', 'kitchen', 'appliance']),
        }
        
        return keywords
    
    def extract_all_features(self, catalog_content):
        """Extract all text features"""
        features = {}
        
        # Brand
        features['brand'] = self.extract_brand(catalog_content)
        
        # Numeric features
        features.update(self.extract_numeric_features(catalog_content))
        
        # Text statistics
        features.update(self.extract_text_stats(catalog_content))
        
        # Category keywords
        features.update(self.extract_category_keywords(catalog_content))
        
        return features


class ImageFeatureExtractor:
    """Extract features from product images"""
    
    def __init__(self):
        pass
    
    def calculate_brightness(self, image_array):
        """Calculate average brightness of image"""
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            brightness = np.mean(image_array)
        else:
            brightness = np.mean(image_array)
        return brightness
    
    def calculate_contrast(self, image_array):
        """Calculate image contrast"""
        return np.std(image_array)
    
    def detect_dominant_colors(self, image_array, n_colors=3):
        """Detect dominant colors in image"""
        # Simplified color detection
        if len(image_array.shape) == 3:
            # For RGB images
            r_mean = np.mean(image_array[:,:,0])
            g_mean = np.mean(image_array[:,:,1])
            b_mean = np.mean(image_array[:,:,2])
            return [r_mean, g_mean, b_mean]
        return [0, 0, 0]
    
    def extract_image_stats(self, image_array):
        """Extract statistical features from image"""
        return {
            'brightness': self.calculate_brightness(image_array),
            'contrast': self.calculate_contrast(image_array),
            'sharpness': np.std(np.gradient(image_array)),
        }


def create_engineered_features(df, is_test=False):
    """
    Create engineered features for the entire dataframe
    
    Args:
        df: DataFrame with catalog_content
        is_test: Whether this is test data
        
    Returns:
        DataFrame with additional engineered features
    """
    print("Creating engineered features...")
    
    text_extractor = TextFeatureExtractor()
    
    # Extract features for each row
    features_list = []
    for idx, row in df.iterrows():
        features = text_extractor.extract_all_features(row['catalog_content'])
        features['sample_id'] = row['sample_id']
        features_list.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Merge with original dataframe
    df_with_features = df.merge(features_df, on='sample_id', how='left')
    
    print(f"Created {len(features_df.columns)} new features")
    
    return df_with_features


def extract_ipq_from_catalog(catalog_content):
    """
    Extract Item Pack Quantity (IPQ) from catalog content
    This is a key feature for pricing
    """
    patterns = [
        r'pack\s+of\s+(\d+)',
        r'(\d+)\s+pack',
        r'quantity[:\s]+(\d+)',
        r'count[:\s]+(\d+)',
        r'ipq[:\s]+(\d+)',
        r'set\s+of\s+(\d+)',
        r'(\d+)\s+piece',
        r'(\d+)\s+units?',
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


if __name__ == "__main__":
    # Test feature extraction
    sample_text = "Apple iPhone 14 Pro Max 256GB - Pack of 1 - Deep Purple"
    
    extractor = TextFeatureExtractor()
    features = extractor.extract_all_features(sample_text)
    
    print("Extracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    ipq = extract_ipq_from_catalog(sample_text)
    print(f"\nIPQ: {ipq}")