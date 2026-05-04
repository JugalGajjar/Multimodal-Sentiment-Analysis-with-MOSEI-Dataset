import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioSentimentModel(nn.Module):
    """
    A simple feedforward neural network for audio-based sentiment prediction.

    Args:
        input_dim (int): Dimension of input audio features.
        hidden_dim (int, optional): Number of neurons in the first hidden layer. Default is 128.
        dropout_rate (float, optional): Dropout rate for regularization. Default is 0.3.

    Forward Input:
        x (Tensor): Audio feature tensor of shape (batch_size, input_dim)

    Forward Output:
        Tensor: Predicted sentiment score of shape (batch_size, 1)
    """
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super(AudioSentimentModel, self).__init__()
        
        # First fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)

        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)

        # Output layer for regression (sentiment score)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass of the feedforward sentiment model.

        Args:
            x (Tensor): Audio features (batch_size, input_dim)

        Returns:
            Tensor: Predicted sentiment score (batch_size, 1)
        """
        # First fully connected layer with batch normalization and ReLU
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second fully connected layer with batch normalization and ReLU
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x

class TransformerAudioEncoder(nn.Module):
    """
    Transformer-based encoder for audio features with sentiment prediction.

    Args:
        input_dim (int): Dimension of input audio features.
        hidden_dim (int, optional): Hidden size for transformer encoder. Default is 128.
        num_layers (int, optional): Number of transformer encoder layers. Default is 2.
        num_heads (int, optional): Number of attention heads in each transformer layer. Default is 4.
        dropout_rate (float, optional): Dropout rate for regularization. Default is 0.3.

    Forward Input:
        x (Tensor): Audio feature tensor of shape (batch_size, input_dim)

    Forward Output:
        Tuple[Tensor, Tensor]: 
            - Encoded audio representation (batch_size, hidden_dim)
            - Sentiment score (batch_size, 1)
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_heads=4, dropout_rate=0.3):
        super(TransformerAudioEncoder, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation="relu",
            batch_first=True
        )

        # Stack multiple transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def get_encoded_features(self, x):
        """
        Extract high-level representation of audio input using transformer encoder.

        Args:
            x (Tensor): Audio input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: Encoded representation of shape (batch_size, hidden_dim)
        """
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add batch dimension if needed (for transformer)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Use first token's representation as the summary
        encoded = x[:, 0, :]
        
        return encoded
    
    def forward(self, x):
        """
        Forward pass through transformer encoder and sentiment head.

        Args:
            x (Tensor): Audio feature tensor of shape (batch_size, input_dim)

        Returns:
            Tuple[Tensor, Tensor]: 
                - Encoded audio representation (batch_size, hidden_dim)
                - Sentiment prediction (batch_size, 1)
        """
        # Get encoded features
        encoded = self.get_encoded_features(x)
        
        # Project to sentiment score
        sentiment = self.output_projection(encoded)
        
        return encoded, sentiment