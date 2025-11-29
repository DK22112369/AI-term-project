import torch
import torch.nn as nn
import torch.nn.functional as F

class TabTransformer(nn.Module):
    """
    TabTransformer: A Transformer-based architecture for tabular data.
    
    References:
    - Huang et al. "TabTransformer: Tabular Data Modeling Using Contextual Embeddings". arXiv:2012.06678
    
    This implementation adapts the architecture for the Crash Severity Prediction task:
    1. Categorical features are embedded and passed through a Transformer Encoder.
    2. Numerical features are normalized (assumed preprocessed) and concatenated.
    3. The output of the Transformer (contextual embeddings) is flattened and concatenated with numerical features.
    4. A final MLP predicts the target.
    """
    def __init__(self, 
                 cat_cardinalities, 
                 num_continuous, 
                 num_classes, 
                 dim_embedding=32, 
                 depth=2, 
                 heads=4, 
                 dim_head=16, 
                 mlp_hidden_mults=(4, 2), 
                 dropout=0.1,
                 use_ohe_input=False):
        """
        Args:
            cat_cardinalities (list of int): 
                If use_ohe_input=False: List of cardinalities (unique values) for each feature.
                If use_ohe_input=True: List of input dimensions (OHE size) for each feature group.
            num_continuous (int): Number of continuous features.
            num_classes (int): Number of output classes.
            dim_embedding (int): Dimension of the embedding.
            depth (int): Number of Transformer layers.
            heads (int): Number of heads.
            dim_head (int): Dimension of head.
            mlp_hidden_mults (tuple): MLP hidden multipliers.
            dropout (float): Dropout.
            use_ohe_input (bool): If True, uses Linear layers to project OHE inputs to embeddings.
        """
        super(TabTransformer, self).__init__()
        
        self.num_continuous = num_continuous
        self.dim_embedding = dim_embedding
        self.use_ohe_input = use_ohe_input
        
        # 1. Embeddings
        if self.use_ohe_input:
            # Use Linear layers to project OHE vectors to embedding space
            self.embeddings = nn.ModuleList([
                nn.Linear(card, dim_embedding) for card in cat_cardinalities
            ])
        else:
            # Standard Lookup Embeddings
            self.embeddings = nn.ModuleList([
                nn.Embedding(card, dim_embedding) for card in cat_cardinalities
            ])
        
        num_cat = len(cat_cardinalities)
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding, 
            nhead=heads, 
            dim_feedforward=dim_embedding * 4, 
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 3. MLP Head
        input_dim = (num_cat * dim_embedding) + num_continuous
        
        hidden_dims = [input_dim * mult for mult in mlp_hidden_mults]
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
        
        self.norm_cont = nn.LayerNorm(num_continuous) if num_continuous > 0 else nn.Identity()

    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat: 
                If use_ohe_input=False: (Batch, Num_Cat) LongTensor
                If use_ohe_input=True: List of (Batch, OHE_Dim) Tensors (one per feature)
            x_cont: (Batch, Num_Continuous) FloatTensor
        """
        # 1. Embed
        embeddings = []
        if self.use_ohe_input:
            # x_cat is expected to be a list of tensors or a split tensor
            # For simplicity, let's assume x_cat is a list of tensors corresponding to each categorical feature group
            for i, emb_layer in enumerate(self.embeddings):
                embeddings.append(emb_layer(x_cat[i]))
        else:
            for i, emb_layer in enumerate(self.embeddings):
                embeddings.append(emb_layer(x_cat[:, i]))
            
        # Stack -> (B, N_cat, Dim)
        x = torch.stack(embeddings, dim=1)
        
        # 2. Transformer
        x = self.transformer(x)
        
        # 3. Flatten
        x = x.flatten(1)
        
        # 4. Concat
        if self.num_continuous > 0:
            x_cont = self.norm_cont(x_cont)
            x = torch.cat((x, x_cont), dim=1)
            
        # 5. MLP
        return self.mlp(x)
