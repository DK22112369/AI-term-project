import torch
import torch.nn as nn

class EarlyFusionMLP(nn.Module):
    """
    Early Fusion MLP for Traffic Accident Severity Prediction.
    
    Unlike CrashSeverityNet (Late Fusion), this model concatenates all input features
    (Driver, Environment, Time/Location) into a single vector at the beginning
    and processes them through a unified MLP.
    
    Architecture:
        Input (Concat of all features)
        Linear -> ReLU -> Dropout
        Linear -> ReLU -> Dropout
        Linear -> ReLU -> Dropout
        Linear (Output)
    """
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        """
        Args:
            input_dim (int): Total dimension of all input features combined.
            num_classes (int): Number of output classes.
            hidden_dims (list): List of hidden layer sizes.
            dropout_rate (float): Dropout probability.
        """
        super(EarlyFusionMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Concatenated input features.
            
        Returns:
            torch.Tensor: Logits.
        """
        return self.mlp(x)
