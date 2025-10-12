import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

import config
from utils import download_images
import pandas as pd


def download_all_images():
    """Download all product images"""
    print("="*50)
    print("Step 1: Downloading Images")
    print("="*50)
    
    # Load datasets
    train_df = pd.read_csv(config.TRAIN_FILE)
    test_df = pd.read_csv(config.TEST_FILE)
    
    # Combine image links
    all_image_links = list(train_df['image_link']) + list(test_df['image_link'])
    all_image_links = list(set(all_image_links))  # Remove duplicates
    
    print(f"Total unique images to download: {len(all_image_links)}")
    
    # Download images
    download_images(all_image_links, str(config.IMAGES_DIR))
    
    print(f"\n✓ Images downloaded to: {config.IMAGES_DIR}")
    print(f"Total images: {len(os.listdir(config.IMAGES_DIR))}")


def train_model():
    """Train the pricing model"""
    print("\n" + "="*50)
    print("Step 2: Training Model")
    print("="*50)
    
    from train import main as train_main
    train_main()


def generate_predictions():
    """Generate predictions on test set"""
    print("\n" + "="*50)
    print("Step 3: Generating Predictions")
    print("="*50)
    
    from predict import main as predict_main
    predict_main()


def run_eda():
    """Run exploratory data analysis"""
    print("\n" + "="*50)
    print("Running Exploratory Data Analysis")
    print("="*50)
    
    from data_loader import load_data, analyze_price_distribution, get_text_length_stats
    
    train_df, val_df, test_df = load_data()
    
    # Analyze price distribution
    analyze_price_distribution(train_df)
    
    # Analyze text
    text_stats = get_text_length_stats(train_df)
    print(f"\nText Statistics: {text_stats}")
    
    # Sample data
    print("\nSample training data:")
    print(train_df.head())


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='ML Challenge 2025 - Product Pricing')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['download', 'train', 'predict', 'eda', 'all'],
                       help='Execution mode')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip image download (if already downloaded)')
    
    args = parser.parse_args()
    
    print("="*50)
    print("ML CHALLENGE 2025 - SMART PRODUCT PRICING")
    print("="*50)
    print(f"Mode: {args.mode}")
    print(f"Device: {'CUDA' if config.DEVICE == 'cuda' else 'CPU'}")
    print("="*50)
    
    try:
        if args.mode == 'download':
            download_all_images()
        
        elif args.mode == 'train':
            if not args.skip_download:
                download_all_images()
            train_model()
        
        elif args.mode == 'predict':
            generate_predictions()
        
        elif args.mode == 'eda':
            run_eda()
        
        elif args.mode == 'all':
            # Full pipeline
            if not args.skip_download:
                download_all_images()
            train_model()
            generate_predictions()
        
        print("\n" + "="*50)
        print("✓ EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        if args.mode in ['predict', 'all']:
            print(f"\n📄 Submission file ready: {config.OUTPUT_FILE}")
            print("Upload this file to the competition portal.")
    
    except Exception as e:
        print("\n" + "="*50)
        print("✗ ERROR OCCURRED")
        print("="*50)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()