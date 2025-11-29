import random
import os
import numpy as np
import torch

def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    Also ensures deterministic behavior in CuDNN.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    print(f"Seed set to {seed} for reproducibility.")
