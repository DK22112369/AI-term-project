import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    """
    A reusable Multilayer Perceptron (MLP) block for processing individual feature groups.
    
    This block is used to encode specific feature sets (Driver, Environment, Time/Location)
    into a latent representation before fusion.
    
    Architecture:
        Linear(in_dim -> hidden_size)
        ReLU
        Dropout(dropout_rate)
        Linear(hidden_size -> hidden_size)
        ReLU
        Dropout(dropout_rate)
    """
    def __init__(self, in_dim, hidden_size=64, dropout_rate=0.3):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return x

class CrashSeverityNet(nn.Module):
    """
    A Multi-Input Deep Learning Model for Traffic Accident Severity Prediction.
    
    This model uses a Late Fusion architecture to process three distinct groups of features:
    1. Driver/Infrastructure Features (processed by d_block)
    2. Environment/Weather Features (processed by e_block)
    3. Time/Location Features (processed by t_block)
    
    The outputs of these three encoders are concatenated and passed through a Fusion MLP
    to predict the final accident severity class.
    """
    def __init__(self, d_in, e_in, t_in, num_classes, block_hidden_size=64):
        """
        Args:
            d_in (int): Input dimension for Driver features.
            e_in (int): Input dimension for Environment features.
            t_in (int): Input dimension for Time/Location features.
            num_classes (int): Number of output classes (severity levels).
            block_hidden_size (int): Hidden size for the individual MLP blocks.
        """
        super(CrashSeverityNet, self).__init__()
        
        # 1. Driver/Infrastructure Encoder
        # Processes features like Distance, Traffic_Signal, Junction, Crossing, etc.
        self.d_block = MLPBlock(d_in, hidden_size=block_hidden_size)
        
        # 2. Environment/Weather Encoder
        # Processes features like Temperature, Humidity, Visibility, Weather_Condition, etc.
        self.e_block = MLPBlock(e_in, hidden_size=block_hidden_size)
        
        # 3. Time/Location Encoder
        # Processes features like Hour, DayOfWeek, Month, Sunrise_Sunset, Duration, etc.
        self.t_block = MLPBlock(t_in, hidden_size=block_hidden_size)
        
        # 4. Fusion Layer
        # Concatenates the 3 latent vectors and predicts severity.
        fusion_in = block_hidden_size * 3
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, xd, xe, xt):
        """
        Forward pass of the model.
        
        Args:
            xd (torch.Tensor): Driver feature batch.
            xe (torch.Tensor): Environment feature batch.
            xt (torch.Tensor): Time/Location feature batch.
            
        Returns:
            torch.Tensor: Logits for each class.
        """
        # Encode each feature group independently
        d_out = self.d_block(xd)
        e_out = self.e_block(xe)
        t_out = self.t_block(xt)
        
        # Fusion: Concatenate along the feature dimension (dim=1)
        concat = torch.cat((d_out, e_out, t_out), dim=1)
        
        # Final prediction
        return self.fusion_mlp(concat)
