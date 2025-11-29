import torch
import os
import sys
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.crash_severity_net import CrashSeverityNet
from data.preprocess_us_accidents import transform_with_preprocessors, clean_and_engineer_features, load_full_dataset

def inspect_model(model_path):
    print(f"Inspecting {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        keys = list(state_dict.keys())
        print(f"  Keys ({len(keys)}): {keys[:5]}...")
        return state_dict
    except Exception as e:
        print(f"  Error loading: {e}")
        return None

def evaluate_specific(model_path, prep_path, df_test):
    print(f"Evaluating {model_path} with {prep_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(prep_path):
        print("  Preprocessor not found.")
        return

    try:
        bundle = joblib.load(prep_path)
        X_t, X_w, X_r, X_s, y = transform_with_preprocessors(df_test, bundle, device=device)
    except Exception as e:
        print(f"  Prep Error: {e}")
        return

    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Infer dims
        input_dims = {}
        # Try various key patterns
        if "encoders.temporal.net.0.weight" in state_dict:
            input_dims['temporal'] = state_dict["encoders.temporal.net.0.weight"].shape[1]
        elif "encoders.temporal.0.weight" in state_dict:
             input_dims['temporal'] = state_dict["encoders.temporal.0.weight"].shape[1]
             
        if "encoders.weather.net.0.weight" in state_dict:
            input_dims['weather'] = state_dict["encoders.weather.net.0.weight"].shape[1]
        elif "encoders.weather.0.weight" in state_dict:
            input_dims['weather'] = state_dict["encoders.weather.0.weight"].shape[1]
            
        if "encoders.road.net.0.weight" in state_dict:
            input_dims['road'] = state_dict["encoders.road.net.0.weight"].shape[1]
        elif "encoders.road.0.weight" in state_dict:
            input_dims['road'] = state_dict["encoders.road.0.weight"].shape[1]
            
        if "encoders.spatial.net.0.weight" in state_dict:
            input_dims['spatial'] = state_dict["encoders.spatial.net.0.weight"].shape[1]
        elif "encoders.spatial.0.weight" in state_dict:
            input_dims['spatial'] = state_dict["encoders.spatial.0.weight"].shape[1]
            
        print(f"  Inferred Dims: {input_dims}")
        
        if not input_dims:
            print("  Could not infer dims.")
            return

        num_classes = 4
        model = CrashSeverityNet(input_dims, num_classes).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        dataset = TensorDataset(X_t, X_w, X_r, X_s, y)
        loader = DataLoader(dataset, batch_size=256)
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for xt, xw, xr, xs, target in loader:
                inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        acc = accuracy_score(all_targets, all_preds)
        recalls = recall_score(all_targets, all_preds, average=None, labels=[0,1,2,3], zero_division=0)
        sev4_recall = recalls[3] if len(recalls) > 3 else 0.0
        
        print(f"  Accuracy: {acc:.4f}, Sev 4 Recall: {sev4_recall:.4f}")
        
    except Exception as e:
        print(f"  Eval Error: {e}")

def main():
    # Load Data
    if os.path.exists("US_Accidents_small.csv"):
        df = load_full_dataset("US_Accidents_small.csv")
    else:
        df = pd.read_csv("data/raw/US_Accidents_March23.csv", nrows=10000)
    df = clean_and_engineer_features(df)
    df_test = df.sample(2000, random_state=42) if len(df) > 2000 else df
    
    # 1. Inspect crash_severity_net_ce_time.pt
    inspect_model("models/crash_severity_net_ce_time.pt")
    
    # 2. Evaluate crash_severity_net_ce_weighted.pt with crash_severity_net_ce_time_preprocessors.joblib
    evaluate_specific("models/crash_severity_net_ce_weighted.pt", "models/crash_severity_net_ce_time_preprocessors.joblib", df_test)
    
    # 3. Evaluate crash_severity_net_ce_time.pt with its own preprocessor
    evaluate_specific("models/crash_severity_net_ce_time.pt", "models/crash_severity_net_ce_time_preprocessors.joblib", df_test)

if __name__ == "__main__":
    main()
