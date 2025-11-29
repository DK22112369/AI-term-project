import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crash_severity_net import CrashSeverityNet

def generate_weights_plot():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Model
    # We need input dims to initialize model.
    # We can infer them from the preprocessor or just hardcode if we know them.
    # Or load the preprocessor.
    prep_path = "models/crash_severity_net_focal_time_preprocessors.joblib"
    if not os.path.exists(prep_path):
        print("Preprocessor not found.")
        return
        
    bundle = joblib.load(prep_path)
    
    # Infer dims from preprocessors
    # This is tricky without data.
    # But we can try to load the model with "safe" dims and then load state dict?
    # No, state dict loading checks shapes.
    
    # Let's load the model state dict and infer dims from it!
    model_path = "models/crash_severity_net_focal_time.pt"
    state_dict = torch.load(model_path, map_location=device)
    
    input_dims = {}
    if "encoders.temporal.net.0.weight" in state_dict:
        input_dims['temporal'] = state_dict["encoders.temporal.net.0.weight"].shape[1]
    if "encoders.weather.net.0.weight" in state_dict:
        input_dims['weather'] = state_dict["encoders.weather.net.0.weight"].shape[1]
    if "encoders.road.net.0.weight" in state_dict:
        input_dims['road'] = state_dict["encoders.road.net.0.weight"].shape[1]
    if "encoders.spatial.net.0.weight" in state_dict:
        input_dims['spatial'] = state_dict["encoders.spatial.net.0.weight"].shape[1]
        
    print(f"Inferred input dims: {input_dims}")
    
    num_classes = 4
    model = CrashSeverityNet(input_dims, num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    feature_names = []
    importances = []
    
    # Extract weights
    for group in ['temporal', 'weather', 'road', 'spatial']:
        if group in model.encoders:
            # Access the first Linear layer
            # model.encoders[group] is MLPBlock
            # MLPBlock.net is Sequential
            # [0] is Linear
            w = model.encoders[group].net[0].weight.detach().cpu().numpy()
            # Mean absolute weight
            imp = np.mean(np.abs(w), axis=0)
            importances.extend(imp)
            feature_names.extend([f"{group}_{i}" for i in range(len(imp))])
            
    # Plot Top 20
    plt.figure(figsize=(10, 8))
    indices = np.argsort(importances)[::-1][:20]
    plt.barh(range(20), np.array(importances)[indices], align='center')
    plt.yticks(range(20), np.array(feature_names)[indices])
    plt.xlabel('Mean Absolute Weight')
    plt.title('Feature Importance (Model Weights)')
    plt.gca().invert_yaxis()
    
    os.makedirs("thesis_materials/figures", exist_ok=True)
    save_path = "thesis_materials/figures/shap_summary_dot.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved fallback feature importance to {save_path}")

if __name__ == "__main__":
    generate_weights_plot()
