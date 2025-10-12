import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ViTImageProcessor
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import config
from data_loader import ProductDataset, load_data
from model import get_model


class AdaptiveEnsemble:
    """
    Adaptive Ensemble that weights models differently based on price range
    Innovation: Different models perform better in different price ranges
    """
    
    def __init__(self, models, device):
        """
        Args:
            models: List of trained PyTorch models
            device: torch device
        """
        self.models = models
        self.device = device
        self.price_range_weights = {}
        
        # Initialize with uniform weights
        for price_range in config.PRICE_RANGES:
            self.price_range_weights[price_range] = [1.0 / len(models)] * len(models)
    
    def get_price_range(self, price):
        """Get price range for a given price"""
        for price_range in config.PRICE_RANGES:
            if price_range[0] <= price < price_range[1]:
                return price_range
        return config.PRICE_RANGES[-1]
    
    def calibrate_weights(self, val_loader):
        """
        Calibrate ensemble weights on validation set
        Uses Ridge regression to find optimal weights per price range
        """
        print("Calibrating ensemble weights...")
        
        # Collect predictions from all models
        all_model_preds = [[] for _ in range(len(self.models))]
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Collecting predictions'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image = batch['image'].to(self.device)
                ipq = batch['ipq'].to(self.device)
                price = batch['price'].cpu().numpy()
                
                # Get predictions from each model
                for model_idx, model in enumerate(self.models):
                    model.eval()
                    pred, _ = model(input_ids, attention_mask, image, ipq)
                    pred = pred.squeeze().cpu().numpy()
                    
                    if pred.shape == ():
                        pred = [pred.item()]
                    
                    all_model_preds[model_idx].extend(pred)
                
                all_targets.extend(price)
        
        # Convert to numpy arrays
        all_model_preds = np.array(all_model_preds).T  # Shape: (n_samples, n_models)
        all_targets = np.array(all_targets)
        
        # Inverse log transform if needed
        if config.PRICE_LOG_TRANSFORM:
            all_model_preds = np.expm1(all_model_preds)
            all_targets = np.expm1(all_targets)
        
        # Calibrate weights for each price range
        for price_range in config.PRICE_RANGES:
            # Get samples in this price range
            mask = (all_targets >= price_range[0]) & (all_targets < price_range[1])
            
            if mask.sum() < 10:  # Skip if too few samples
                continue
            
            X = all_model_preds[mask]
            y = all_targets[mask]
            
            # Use Ridge regression to find optimal weights
            ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False)
            ridge.fit(X, y)
            
            # Normalize weights to sum to 1
            weights = ridge.coef_
            weights = weights / weights.sum()
            
            self.price_range_weights[price_range] = weights
            
            print(f"Price range {price_range}: weights = {weights}")
    
    def predict(self, test_loader):
        """Generate ensemble predictions"""
        all_predictions = []
        all_sample_ids = []
        
        print("Generating ensemble predictions...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Ensemble prediction'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image = batch['image'].to(self.device)
                ipq = batch['ipq'].to(self.device)
                sample_ids = batch['sample_id']
                
                # Get predictions from all models
                model_preds = []
                for model in self.models:
                    model.eval()
                    pred, _ = model(input_ids, attention_mask, image, ipq)
                    pred = pred.squeeze().cpu().numpy()
                    
                    if pred.shape == ():
                        pred = np.array([pred.item()])
                    
                    model_preds.append(pred)
                
                model_preds = np.array(model_preds).T  # Shape: (batch_size, n_models)
                
                # Inverse log transform
                if config.PRICE_LOG_TRANSFORM:
                    model_preds = np.expm1(model_preds)
                
                # Weighted average based on initial estimate
                batch_preds = []
                for sample_preds in model_preds:
                    # Use simple average as initial estimate
                    initial_pred = np.mean(sample_preds)
                    
                    # Get weights for this price range
                    price_range = self.get_price_range(initial_pred)
                    weights = self.price_range_weights.get(
                        price_range, 
                        [1.0 / len(self.models)] * len(self.models)
                    )
                    
                    # Weighted prediction
                    final_pred = np.sum(sample_preds * weights)
                    batch_preds.append(final_pred)
                
                all_predictions.extend(batch_preds)
                all_sample_ids.extend(sample_ids)
        
        return all_sample_ids, np.array(all_predictions)


class SimpleEnsemble:
    """Simple weighted average ensemble"""
    
    def __init__(self, models, weights, device):
        """
        Args:
            models: List of trained PyTorch models
            weights: List of weights for each model (should sum to 1)
            device: torch device
        """
        self.models = models
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()  # Normalize
        self.device = device
    
    def predict(self, test_loader):
        """Generate ensemble predictions"""
        all_predictions = []
        all_sample_ids = []
        
        print(f"Generating predictions with weights: {self.weights}")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Ensemble prediction'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image = batch['image'].to(self.device)
                ipq = batch['ipq'].to(self.device)
                sample_ids = batch['sample_id']
                
                # Get predictions from all models
                batch_preds = None
                for model_idx, model in enumerate(self.models):
                    model.eval()
                    pred, _ = model(input_ids, attention_mask, image, ipq)
                    pred = pred.squeeze().cpu().numpy()
                    
                    if pred.shape == ():
                        pred = np.array([pred.item()])
                    
                    # Inverse log transform
                    if config.PRICE_LOG_TRANSFORM:
                        pred = np.expm1(pred)
                    
                    # Weighted sum
                    if batch_preds is None:
                        batch_preds = pred * self.weights[model_idx]
                    else:
                        batch_preds += pred * self.weights[model_idx]
                
                all_predictions.extend(batch_preds)
                all_sample_ids.extend(sample_ids)
        
        return all_sample_ids, np.array(all_predictions)


def train_multiple_models():
    """
    Train multiple models with different architectures/seeds
    This creates diversity for ensembling
    """
    print("Training multiple models for ensemble...")
    
    # Model configurations
    model_configs = [
        {'type': 'multimodal', 'name': 'multimodal_model_1'},
        {'type': 'text_only', 'name': 'text_only_model'},
        {'type': 'image_only', 'name': 'image_only_model'},
    ]
    
    trained_models = []
    
    for config_dict in model_configs:
        print(f"\nTraining {config_dict['name']}...")
        # Training code would go here
        # For now, this is a placeholder
        pass
    
    return trained_models


if __name__ == "__main__":
    print("Ensemble module loaded successfully!")
    print(f"Available ensemble methods:")
    print("  - AdaptiveEnsemble: Adaptive weights per price range")
    print("  - SimpleEnsemble: Fixed weighted average")