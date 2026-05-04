import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism for computing attention between different modalities.
    
    Args:
        query_dim (int): Dimension of the query input.
        key_dim (int): Dimension of the key input.
        value_dim (int): Dimension of the value input.
        hidden_dim (int): Internal hidden dimension used for multi-head attention.
        num_heads (int): Number of attention heads.
        dropout_rate (float): Dropout rate for regularization.
    """
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, num_heads=8, dropout_rate=0.1):
        super(CrossAttention, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Linear layers to project query, key, and value to hidden dimensions
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, query, key, value):
        """
        Forward pass for cross-attention.
        
        Args:
            query (Tensor): Query tensor of shape (batch_size, seq_len_q, query_dim).
            key (Tensor): Key tensor of shape (batch_size, seq_len_k, key_dim).
            value (Tensor): Value tensor of shape (batch_size, seq_len_v, value_dim).
            
        Returns:
            Tensor: Attention output of shape (batch_size, seq_len_q, hidden_dim) or (batch_size, hidden_dim) if seq_len_q = 1.
        """
        batch_size = query.size(0)
        
        # Project query, key, value
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, value)
        
        # Reshape and project output
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.output_proj(attended_values)
        
        # Remove sequence dimension if only one token
        if output.size(1) == 1:
            output = output.squeeze(1)
        
        return output

class MultimodalCrossAttention(nn.Module):
    """
    Multimodal fusion using cross-attention between text, audio, and visual features.
    
    Args:
        text_dim (int): Dimension of text input features.
        audio_dim (int): Dimension of audio input features.
        visual_dim (int): Dimension of visual input features.
        hidden_dim (int): Hidden dimension to which all modalities are projected.
        num_heads (int): Number of attention heads.
        dropout_rate (float): Dropout rate for regularization.
    """
    def __init__(
        self, 
        text_dim, 
        audio_dim, 
        visual_dim, 
        hidden_dim=256, 
        num_heads=8, 
        dropout_rate=0.1
    ):
        super(MultimodalCrossAttention, self).__init__()
        
        # Project input modalities to common hidden dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        
        # Cross-attention modules
        # Text as query, attend to audio and visual
        self.text_audio_attn = CrossAttention(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads, dropout_rate
        )
        self.text_visual_attn = CrossAttention(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads, dropout_rate
        )
        
        # Audio as query, attend to text and visual
        self.audio_text_attn = CrossAttention(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads, dropout_rate
        )
        self.audio_visual_attn = CrossAttention(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads, dropout_rate
        )
        
        # Visual as query, attend to text and audio
        self.visual_text_attn = CrossAttention(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads, dropout_rate
        )
        self.visual_audio_attn = CrossAttention(
            hidden_dim, hidden_dim, hidden_dim, hidden_dim, num_heads, dropout_rate
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output projection for sentiment prediction
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, text_features, audio_features, visual_features):
        """
        Forward pass for multimodal cross-attention fusion.
        
        Args:
            text_features (Tensor): Text features of shape (batch_size, text_dim).
            audio_features (Tensor): Audio features of shape (batch_size, audio_dim).
            visual_features (Tensor): Visual features of shape (batch_size, visual_dim).
        
        Returns:
            fused_features (Tensor): Multimodal fused features (batch_size, hidden_dim).
            sentiment (Tensor): Sentiment prediction score (batch_size, 1).
        """
        batch_size = text_features.size(0)
        
        # Project features to the hidden dimension
        text_proj = self.text_proj(text_features)
        audio_proj = self.audio_proj(audio_features)
        visual_proj = self.visual_proj(visual_features)
        
        # Add sequence dimension for attention if needed (e.g., [B, D] -> [B, 1, D])
        if len(text_proj.shape) == 2:
            text_proj = text_proj.unsqueeze(1)
            audio_proj = audio_proj.unsqueeze(1)
            visual_proj = visual_proj.unsqueeze(1)
        
        # Cross-attention: text attending to other modalities
        text_audio_attn = self.text_audio_attn(text_proj, audio_proj, audio_proj)
        text_visual_attn = self.text_visual_attn(text_proj, visual_proj, visual_proj)
        text_context = text_proj + text_audio_attn + text_visual_attn
        
        # Cross-attention: audio attending to other modalities
        audio_text_attn = self.audio_text_attn(audio_proj, text_proj, text_proj)
        audio_visual_attn = self.audio_visual_attn(audio_proj, visual_proj, visual_proj)
        audio_context = audio_proj + audio_text_attn + audio_visual_attn
        
        # Cross-attention: visual attending to other modalities
        visual_text_attn = self.visual_text_attn(visual_proj, text_proj, text_proj)
        visual_audio_attn = self.visual_audio_attn(visual_proj, audio_proj, audio_proj)
        visual_context = visual_proj + visual_text_attn + visual_audio_attn
        
        # Remove sequence dimension ([B, 1, D] -> [B, D])
        if len(text_proj.shape) == 3:
            text_proj = text_proj.squeeze(1)
            audio_proj = audio_proj.squeeze(1)
            visual_proj = visual_proj.squeeze(1)
        
        # Concatenate fused features from all modalities
        concat_features = torch.cat([text_features, audio_features, visual_features], dim=1)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(concat_features)
        
        # Predict sentiment
        sentiment = self.output_proj(fused_features)
        
        return fused_features, sentiment