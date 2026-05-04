import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyFusionModel(nn.Module):
    """
    Early fusion model for multimodal sentiment analysis.
    Concatenates features from text, audio, and visual modalities,
    then applies fully connected layers for sentiment prediction.
    Args:
        text_dim (int): Dimensionality of text features.
        audio_dim (int): Dimensionality of audio features.
        visual_dim (int): Dimensionality of visual features.
        hidden_dim (int): Hidden layer size for fusion MLP.
        dropout_rate (float): Dropout rate for regularization.
    """
    def __init__(
        self, 
        text_dim, 
        audio_dim, 
        visual_dim, 
        hidden_dim=256, 
        dropout_rate=0.3
    ):
        super(EarlyFusionModel, self).__init__()
        
        # Total input dimension after concatenation
        total_dim = text_dim + audio_dim + visual_dim
        
        # Fully connected layers for fusion and prediction
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, text_features, audio_features, visual_features):
        """
        Forward pass for early fusion model.

        Args:
            text_features (Tensor): Tensor of shape [batch_size, text_dim].
            audio_features (Tensor): Tensor of shape [batch_size, audio_dim].
            visual_features (Tensor): Tensor of shape [batch_size, visual_dim].

        Returns:
            Tensor: Predicted sentiment score of shape [batch_size, 1].
        """
        # Concatenate modality features along the feature dimension
        concat_features = torch.cat([text_features, audio_features, visual_features], dim=1)
        
        # Apply fusion layers
        sentiment = self.fusion_layers(concat_features)
        
        return sentiment

class LateFusionModel(nn.Module):
    """
    Late fusion model for multimodal sentiment analysis.
    Uses individual models for each modality and combines
    their predictions using weighted averaging.
    Args:
        text_model (nn.Module): Model for text modality.
        audio_model (nn.Module): Model for audio modality.
        visual_model (nn.Module): Model for visual modality.
        fusion_weights (list, optional): Weights for each modality. If None, weights are learned.
    """
    def __init__(
        self, 
        text_model, 
        audio_model, 
        visual_model, 
        fusion_weights=None
    ):
        super(LateFusionModel, self).__init__()
        
        # Store individual models
        self.text_model = text_model
        self.audio_model = audio_model
        self.visual_model = visual_model
        
        # Freeze parameters of individual models
        for model in [self.text_model, self.audio_model, self.visual_model]:
            for param in model.parameters():
                param.requires_grad = False
        
        # Initialize fusion weights
        if fusion_weights is None:
            # Learn the weights if not provided
            self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        else:
            # Use fixed weights if provided
            self.fusion_weights = torch.tensor(fusion_weights)
    
    def forward(self, text_features, audio_features, visual_features):
        """
        Forward pass for late fusion model.

        Args:
            text_features (Tensor): Input tensor for text model.
            audio_features (Tensor): Input tensor for audio model.
            visual_features (Tensor): Input tensor for visual model.

        Returns:
            Tensor: Weighted sentiment prediction of shape [batch_size, 1].
        """
        # Get predictions from each modality
        text_pred = self.text_model(text_features)
        audio_pred = self.audio_model(audio_features)
        visual_pred = self.visual_model(visual_features)
        
        # Normalize fusion weights using softmax
        weights = F.softmax(self.fusion_weights, dim=0)

        # Compute weighted sum of predictions
        combined_pred = (
            weights[0] * text_pred + 
            weights[1] * audio_pred + 
            weights[2] * visual_pred
        )
        
        return combined_pred

class TransformerFusionModel(nn.Module):
    """
    Transformer-based fusion model for multimodal sentiment analysis.
    Uses Transformer encoders for individual modalities and a cross-attention
    module to combine them into a fused sentiment representation.
    Args:
        text_dim (int): Dimensionality of text features.
        audio_dim (int): Dimensionality of audio features.
        visual_dim (int): Dimensionality of visual features.
        hidden_dim (int): Hidden layer size for transformer encoders.
        num_heads (int): Number of attention heads in transformer layers.
        num_layers (int): Number of transformer encoder layers.
        dropout_rate (float): Dropout rate for regularization.
    """
    def __init__(
        self, 
        text_dim, 
        audio_dim, 
        visual_dim, 
        hidden_dim=256, 
        num_heads=8, 
        num_layers=4, 
        dropout_rate=0.3
    ):
        super(TransformerFusionModel, self).__init__()
        
        # Local imports to avoid circular dependencies
        from src.models.text import TransformerTextEncoder
        from src.models.audio import TransformerAudioEncoder
        from src.models.visual import TransformerVisualEncoder
        from src.models.attention import MultimodalCrossAttention
        
        # Individual modality encoders
        self.text_encoder = TransformerTextEncoder(
            text_dim, hidden_dim, num_layers, num_heads, dropout_rate
        )
        
        self.audio_encoder = TransformerAudioEncoder(
            audio_dim, hidden_dim // 2, num_layers // 2, num_heads // 2, dropout_rate
        )
        
        self.visual_encoder = TransformerVisualEncoder(
            visual_dim, hidden_dim // 2, num_layers // 2, num_heads // 2, dropout_rate
        )
        
        # Multimodal cross-attention fusion
        self.fusion_module = MultimodalCrossAttention(
            hidden_dim, hidden_dim // 2, hidden_dim // 2, hidden_dim, num_heads, dropout_rate
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, features):
        """
        Forward pass for transformer-based fusion model.

        Args:
            text_features (Tensor): Input tensor for text modality.
            audio_features (Tensor): Input tensor for audio modality.
            visual_features (Tensor): Input tensor for visual modality.

        Returns:
            Tensor: Fused sentiment prediction of shape [batch_size, 1].
        """
        text_features = features["language"]
        audio_features = features["acoustic"]
        visual_features = features["visual"]

        # Encode each modality using respective transformer encoders
        text_encoded = self.text_encoder.get_encoded_features(text_features)
        audio_encoded = self.audio_encoder.get_encoded_features(audio_features)
        visual_encoded = self.visual_encoder.get_encoded_features(visual_features)
        
        # Cross-attention fusion
        fused_features, sentiment = self.fusion_module(
            text_encoded, audio_encoded, visual_encoded
        )
        
        return sentiment