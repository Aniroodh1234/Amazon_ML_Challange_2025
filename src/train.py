import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ViTImageProcessor
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import json

import config
from data_loader import ProductDataset, load_data
from model import get_model

class SMAPELoss(nn.Module):
    """Symmetric Mean Absolute Percentage Error Loss"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        """
        Calculate SMAPE loss
        Formula: mean(|pred - actual| / ((|actual| + |pred|) / 2))
        """
        numerator = torch.abs(predictions - targets)
        denominator = (torch.abs(targets) + torch.abs(predictions)) / 2.0
        # Add small epsilon to avoid division by zero
        smape = numerator / (denominator + 1e-8)
        return torch.mean(smape)


def calculate_smape(predictions, targets):
    """Calculate SMAPE metric for evaluation"""
    numerator = np.abs(predictions - targets)
    denominator = (np.abs(targets) + np.abs(predictions)) / 2.0
    smape = np.mean(numerator / (denominator + 1e-8))
    return smape * 100  # Return as percentage


class Trainer:
    """Training manager"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss functions
        self.smape_loss = SMAPELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * config.NUM_EPOCHS
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Training state
        self.best_val_smape = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_smape': []
        }
    
    def get_price_range_label(self, price):
        """Convert price to range label for auxiliary task"""
        for idx, (min_price, max_price) in enumerate(config.PRICE_RANGES):
            if min_price <= price < max_price:
                return idx
        return len(config.PRICE_RANGES) - 1
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            image = batch['image'].to(self.device)
            ipq = batch['ipq'].to(self.device)
            price = batch['price'].to(self.device)
            
            # Forward pass
            price_pred, price_range_logits = self.model(
                input_ids, attention_mask, image, ipq
            )
            
            # Calculate loss
            # Primary loss: SMAPE
            smape_loss = self.smape_loss(price_pred.squeeze(), price)
            
            # Secondary loss: MSE for stability
            mse_loss = self.mse_loss(price_pred.squeeze(), price)
            
            # Combined loss
            loss = 0.7 * smape_loss + 0.3 * mse_loss
            
            # Auxiliary loss: Price range classification
            if config.USE_PRICE_RANGE_CLASSIFIER and price_range_logits is not None:
                # Convert prices to range labels
                if config.PRICE_LOG_TRANSFORM:
                    actual_prices = torch.expm1(price)
                else:
                    actual_prices = price
                
                price_range_labels = torch.tensor([
                    self.get_price_range_label(p.item()) 
                    for p in actual_prices
                ]).to(self.device)
                
                range_loss = self.ce_loss(price_range_logits, price_range_labels)
                loss = loss + 0.1 * range_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image = batch['image'].to(self.device)
                ipq = batch['ipq'].to(self.device)
                price = batch['price'].to(self.device)
                
                # Forward pass
                price_pred, _ = self.model(input_ids, attention_mask, image, ipq)
                
                # Calculate loss
                loss = self.smape_loss(price_pred.squeeze(), price)
                total_loss += loss.item()
                
                # Store predictions and targets
                all_predictions.extend(price_pred.squeeze().cpu().numpy())
                all_targets.extend(price.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Inverse log transform if needed
        if config.PRICE_LOG_TRANSFORM:
            all_predictions = np.expm1(all_predictions)
            all_targets = np.expm1(all_targets)
        
        # Calculate SMAPE
        smape = calculate_smape(all_predictions, all_targets)
        
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, smape
    
    def save_checkpoint(self, epoch, smape):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'smape': smape,
            'training_history': self.training_history
        }
        
        checkpoint_path = config.MODELS_DIR / f'checkpoint_epoch_{epoch}_smape_{smape:.2f}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model separately
        if smape < self.best_val_smape:
            best_model_path = config.MODELS_DIR / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            print(f"Best model updated: {best_model_path}")
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
        for epoch in range(config.NUM_EPOCHS):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_smape = self.validate()
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_smape'].append(val_smape)
            
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val SMAPE: {val_smape:.2f}%")
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, val_smape)
            
            # Early stopping check
            if val_smape < self.best_val_smape:
                self.best_val_smape = val_smape
                self.epochs_without_improvement = 0
                print(f"  ✓ New best SMAPE: {val_smape:.2f}%")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")
            
            if self.epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation SMAPE: {self.best_val_smape:.2f}%")
        
        # Save training history
        # Save training history
        history_path = config.OUTPUTS_DIR / 'training_history.json'
        
        # Convert numpy types to Python types for JSON serialization
        serializable_history = {
            'train_loss': [float(x) for x in self.training_history['train_loss']],
            'val_loss': [float(x) for x in self.training_history['val_loss']],
            'val_smape': [float(x) for x in self.training_history['val_smape']]
        }
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)


def main():
    """Main training function"""
    print("="*50)
    print("ML Challenge 2025 - Product Pricing Model")
    print("="*50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    train_df, val_df, _ = load_data()
    
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
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ProductDataset(train_df, text_tokenizer, image_transform, is_test=False)
    val_dataset = ProductDataset(val_df, text_tokenizer, image_transform, is_test=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model('multimodal').to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer and start training
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.train()


if __name__ == "__main__":

    main()
