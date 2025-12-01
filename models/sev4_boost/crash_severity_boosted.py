import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    """
    Standard MLP Block for Feature Encoders.
    Structure: Linear -> ReLU -> Dropout -> Linear -> ReLU
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

class CrashSeverityBoosted(nn.Module):
    """
    Fail-Safe Optimized Late Fusion Model.
    
    Features:
    - 4 Independent Encoders (Temporal, Weather, Road, Spatial)
    - Deep Fusion Network (256 -> 256 -> 128 -> 64 -> 4)
    - Designed for high representational power to capture subtle fatal crash patterns.
    """
    def __init__(self, input_dims, num_classes=4, block_hidden_size=64):
        """
        Args:
            input_dims (dict): {'temporal': d1, 'weather': d2, 'road': d3, 'spatial': d4}
            num_classes (int): 4
            block_hidden_size (int): 64
        """
        super(CrashSeverityBoosted, self).__init__()
        
        self.encoders = nn.ModuleDict()
        
        # 1. Encoders
        for group_name, dim in input_dims.items():
            self.encoders[group_name] = MLPBlock(dim, hidden_size=block_hidden_size)
            
        # 2. Deep Fusion Network
        # Input dim = 4 groups * 64 hidden size = 256
        fusion_in = block_hidden_size * len(input_dims)
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, inputs):
        """
        inputs: dict of tensors {'temporal': ..., 'weather': ...}
        """
        encoded_features = []
        
        # Ensure consistent order
        for group_name in sorted(self.encoders.keys()):
            if group_name in inputs:
                x = inputs[group_name]
                out = self.encoders[group_name](x)
                encoded_features.append(out)
            else:
                raise ValueError(f"Missing input for group: {group_name}")
        
        # Concatenate
        concat = torch.cat(encoded_features, dim=1)
        
        # Deep Fusion
        return self.fusion_mlp(concat)
