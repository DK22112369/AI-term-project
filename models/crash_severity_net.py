import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    """
    A reusable Multilayer Perceptron (MLP) block for processing individual feature groups.
    
    Architecture:
        Linear(in_dim -> hidden_size)
        ReLU
        BatchNorm1d
        Dropout(dropout_rate)
        Linear(hidden_size -> hidden_size)
        ReLU
        BatchNorm1d
        Dropout(dropout_rate)
    """
    def __init__(self, in_dim, hidden_size=64, dropout_rate=0.3):
        super(MLPBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)

class CrashSeverityNet(nn.Module):
    """
    A Multi-Input Deep Learning Model for Traffic Accident Severity Prediction.
    
    This model uses a Group-wise Late Fusion architecture to process four distinct groups of features:
    1. Temporal Features
    2. Weather Features
    3. Road/Infrastructure Features
    4. Spatial Features
    
    The outputs of these four encoders are concatenated and passed through a Fusion MLP
    to predict the final accident severity class.
    """
    def __init__(self, input_dims, num_classes, block_hidden_size=64):
        """
        Args:
            input_dims (dict): Dictionary mapping group names to input dimensions.
                               e.g., {'temporal': 10, 'weather': 5, ...}
            num_classes (int): Number of output classes (severity levels).
            block_hidden_size (int): Hidden size for the individual MLP blocks.
        """
        super(CrashSeverityNet, self).__init__()
        
        self.encoders = nn.ModuleDict()
        
        # Create an MLP block for each feature group
        for group_name, dim in input_dims.items():
            self.encoders[group_name] = MLPBlock(dim, hidden_size=block_hidden_size)
            
        # Fusion Layer
        # Concatenates the latent vectors from all groups
        fusion_in = block_hidden_size * len(input_dims)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, inputs):
        """
        Forward pass of the model.
        
        Args:
            inputs (dict): Dictionary mapping group names to input tensors.
                           e.g., {'temporal': tensor, 'weather': tensor, ...}
            
        Returns:
            torch.Tensor: Logits for each class.
        """
        encoded_features = []
        
        # Process each group through its corresponding encoder
        # We iterate in a sorted order of keys to ensure consistent concatenation order
        # assuming input_dims keys match inputs keys
        for group_name in sorted(self.encoders.keys()):
            if group_name in inputs:
                x = inputs[group_name]
                out = self.encoders[group_name](x)
                encoded_features.append(out)
            else:
                raise ValueError(f"Missing input for group: {group_name}")
        
        # Fusion: Concatenate along the feature dimension (dim=1)
        concat = torch.cat(encoded_features, dim=1)
        
        # Final prediction
        return self.fusion_mlp(concat)
