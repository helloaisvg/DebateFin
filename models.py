"""
PyTorch LSTM model for growth prediction
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class LSTMGrowthPredictor(nn.Module):
    """
    LSTM model for predicting financial growth metrics
    """
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super(LSTMGrowthPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
        
        Returns:
            Output tensor of shape (batch, output_size)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and fully connected layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output
    
    def predict(self, x: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray:
        """
        Make prediction
        
        Args:
            x: Input array of shape (seq_len, features) or (batch, seq_len, features)
            device: PyTorch device
        
        Returns:
            Prediction array
        """
        self.eval()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert to tensor
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        # Add batch dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        x = x.to(device)
        self.to(device)
        
        with torch.no_grad():
            output = self.forward(x)
        
        return output.cpu().numpy()


def create_lstm_model(input_size: int = 10, hidden_size: int = 64, num_layers: int = 2) -> LSTMGrowthPredictor:
    """
    Create and initialize LSTM model
    
    Args:
        input_size: Number of input features
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
    
    Returns:
        Initialized LSTM model
    """
    model = LSTMGrowthPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    return model


def load_pretrained_model(model_path: Optional[str] = None) -> LSTMGrowthPredictor:
    """
    Load pretrained model (placeholder for future use)
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded model
    """
    model = create_lstm_model()
    
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded pretrained model from {model_path}")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
    
    return model

