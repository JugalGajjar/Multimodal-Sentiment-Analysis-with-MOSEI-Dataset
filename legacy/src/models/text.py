import torch
import torch.nn as nn
import torch.nn.functional as F

class TextSentimentModel(nn.Module):
    """
    A feedforward neural network for text-based sentiment regression.

    Args:
        input_dim (int): Input feature dimension (e.g., text embeddings).
        hidden_dim (int, optional): Number of neurons in the first hidden layer. Default is 256.
        dropout_rate (float, optional): Dropout rate for regularization. Default is 0.3.

    Forward Input:
        x (Tensor): Input text features of shape (batch_size, input_dim)

    Forward Output:
        Tensor: Predicted sentiment score of shape (batch_size, 1)
    """
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.3):
        super(TextSentimentModel, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Batch normalization and dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
    
    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor with shape (batch_size, input_dim)

        Returns:
            Tensor: Output sentiment prediction (batch_size, 1)
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

class TransformerTextEncoder(nn.Module):
    """
    Transformer-based encoder for extracting features from text input and predicting sentiment.

    Args:
        input_dim (int): Dimension of input text features.
        hidden_dim (int, optional): Dimension of the transformer hidden layer. Default is 256.
        num_layers (int, optional): Number of transformer encoder layers. Default is 4.
        num_heads (int, optional): Number of attention heads. Default is 8.
        dropout_rate (float, optional): Dropout rate for regularization. Default is 0.3.

    Forward Input:
        x (Tensor): Text input tensor of shape (batch_size, input_dim)

    Forward Output:
        Tuple[Tensor, Tensor]:
            - Encoded text representation of shape (batch_size, hidden_dim)
            - Sentiment score prediction of shape (batch_size, 1)
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, num_heads=8, dropout_rate=0.3):
        super(TransformerTextEncoder, self).__init__()
        
        # Project input to transformer hidden dimension
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

        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass to get encoded representation and sentiment prediction.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple[Tensor, Tensor]: 
                - Encoded representation (batch_size, hidden_dim)
                - Sentiment score (batch_size, 1)
        """
        # Get encoded features
        encoded = self.get_encoded_features(x)
        
        # Project to sentiment score
        sentiment = self.output_projection(encoded)
        
        return encoded, sentiment
    
    def get_encoded_features(self, x):
        """
        Encodes input text features using a transformer encoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: Encoded features of shape (batch_size, hidden_dim)
        """
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add batch dimension if needed (for transformer)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Extract the encoded representation (use the first token)
        encoded = x[:, 0, :]
        
        return encoded