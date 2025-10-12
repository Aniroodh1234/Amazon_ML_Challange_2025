import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, ViTModel, ViTImageProcessor
import config

class CrossModalAttention(nn.Module):
    """Cross-attention mechanism between text and image features"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, query, key_value):
        """
        Args:
            query: Tensor of shape (batch, seq_len, dim)
            key_value: Tensor of shape (batch, seq_len, dim)
        """
        attn_output, _ = self.attention(query, key_value, key_value)
        return self.norm(query + attn_output)


class MultimodalPricingModel(nn.Module):
    """
    Multimodal model for product price prediction
    Combines text (DeBERTa) and image (ViT) features
    """
    
    def __init__(self):
        super().__init__()
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        
        # Image encoder
        self.image_encoder = ViTModel.from_pretrained(config.IMAGE_MODEL_NAME)
        
        # Feature dimensions
        self.text_dim = config.TEXT_EMBEDDING_DIM
        self.image_dim = config.IMAGE_EMBEDDING_DIM
        
        # Project features to same dimension
        self.text_projection = nn.Linear(self.text_dim, config.HIDDEN_DIM)
        self.image_projection = nn.Linear(self.image_dim, config.HIDDEN_DIM)
        
        # Cross-modal attention (our innovation)
        if config.USE_CROSS_MODAL_ATTENTION:
            self.cross_attention_text_to_image = CrossModalAttention(
                config.HIDDEN_DIM, 
                config.NUM_ATTENTION_HEADS
            )
            self.cross_attention_image_to_text = CrossModalAttention(
                config.HIDDEN_DIM,
                config.NUM_ATTENTION_HEADS
            )
        
        # Additional feature embedding (IPQ)
        self.ipq_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        
        # Fusion layers
        fusion_input_dim = config.HIDDEN_DIM * 2 + 64  # text + image + ipq
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
        )
        
        # Price prediction head
        self.price_head = nn.Linear(128, 1)
        
        # Price range classifier (auxiliary task)
        if config.USE_PRICE_RANGE_CLASSIFIER:
            self.price_range_classifier = nn.Linear(128, len(config.PRICE_RANGES))
    
    def forward(self, input_ids, attention_mask, image, ipq):
        """
        Forward pass
        
        Args:
            input_ids: Text input IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            image: Image tensor (batch, 3, 224, 224)
            ipq: Item Pack Quantity (batch, 1)
        
        Returns:
            price_pred: Predicted price (batch, 1)
            price_range_logits: Price range classification logits (batch, num_ranges)
        """
        # Encode text
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_output.last_hidden_state[:, 0, :]  # CLS token
        text_features = self.text_projection(text_features)
        
        # Encode image
        # Encode image
        image_output = self.image_encoder(
            pixel_values=image,
            interpolate_pos_encoding=True
        )
        image_features = image_output.last_hidden_state[:, 0, :]  # CLS token
        image_features = self.image_projection(image_features)
        
        # Cross-modal attention
        if config.USE_CROSS_MODAL_ATTENTION:
            # Add sequence dimension for attention
            text_seq = text_features.unsqueeze(1)  # (batch, 1, hidden_dim)
            image_seq = image_features.unsqueeze(1)  # (batch, 1, hidden_dim)
            
            # Apply cross-attention
            text_attended = self.cross_attention_text_to_image(text_seq, image_seq)
            image_attended = self.cross_attention_image_to_text(image_seq, text_seq)
            
            # Remove sequence dimension
            text_features = text_attended.squeeze(1)
            image_features = image_attended.squeeze(1)
        
        # Embed IPQ
        ipq_features = self.ipq_embedding(ipq.unsqueeze(1))
        
        # Concatenate all features
        combined_features = torch.cat([
            text_features,
            image_features,
            ipq_features
        ], dim=1)
        
        # Fusion
        fused_features = self.fusion_layers(combined_features)
        
        # Price prediction
        price_pred = self.price_head(fused_features)
        
        # Price range classification (auxiliary task)
        price_range_logits = None
        if config.USE_PRICE_RANGE_CLASSIFIER:
            price_range_logits = self.price_range_classifier(fused_features)
        
        return price_pred, price_range_logits


class TextOnlyModel(nn.Module):
    """Text-only baseline model"""
    
    def __init__(self):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        
        self.regressor = nn.Sequential(
            nn.Linear(config.TEXT_EMBEDDING_DIM, 512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_output.last_hidden_state[:, 0, :]
        price_pred = self.regressor(text_features)
        return price_pred


class ImageOnlyModel(nn.Module):
    """Image-only baseline model"""
    
    def __init__(self):
        super().__init__()
        self.image_encoder = ViTModel.from_pretrained(config.IMAGE_MODEL_NAME)
        
        self.regressor = nn.Sequential(
            nn.Linear(config.IMAGE_EMBEDDING_DIM, 512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, 1)
        )
    
    def forward(self, image):
        image_output = self.image_encoder(pixel_values=image)
        image_features = image_output.last_hidden_state[:, 0, :]
        price_pred = self.regressor(image_features)
        return price_pred


def get_model(model_type='multimodal'):
    """
    Factory function to get model
    
    Args:
        model_type: 'multimodal', 'text_only', or 'image_only'
    
    Returns:
        model: PyTorch model
    """
    if model_type == 'multimodal':
        return MultimodalPricingModel()
    elif model_type == 'text_only':
        return TextOnlyModel()
    elif model_type == 'image_only':
        return ImageOnlyModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model initialization
    print("Testing model architectures...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test multimodal model
    model = MultimodalPricingModel().to(device)
    print(f"\nMultimodal Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with dummy inputs
    batch_size = 2
    dummy_input_ids = torch.randint(0, 1000, (batch_size, 128)).to(device)
    dummy_attention_mask = torch.ones((batch_size, 128)).to(device)
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_ipq = torch.tensor([1.0, 2.0]).to(device)
    
    with torch.no_grad():
        price_pred, price_range_logits = model(
            dummy_input_ids, 
            dummy_attention_mask, 
            dummy_image, 
            dummy_ipq
        )
    
    print(f"Price prediction shape: {price_pred.shape}")
    if price_range_logits is not None:
        print(f"Price range logits shape: {price_range_logits.shape}")
    
    print("\nModel test successful!")