import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.metrics import accuracy_score, recall_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crash_severity_net import CrashSeverityNet
from data.preprocess_us_accidents import load_full_dataset, stratified_sample, clean_and_engineer_features, transform_with_preprocessors, load_preprocessors

def evaluate_model(model_path, df_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Determine Preprocessor Path
    # Assumption: preprocessors are saved as {model_name}_preprocessors.joblib
    base_name = os.path.splitext(model_path)[0]
    prep_path = f"{base_name}_preprocessors.joblib"
    
    if not os.path.exists(prep_path):
        # Try default or skip
        # print(f"  Preprocessor not found: {prep_path}")
        return None, None, "No Preprocessor"
        
    # 2. Load Preprocessor & Transform
    try:
        bundle = joblib.load(prep_path)
        X_t, X_w, X_r, X_s, y = transform_with_preprocessors(df_test, bundle, device=device)
    except Exception as e:
        return None, None, f"Prep Error: {str(e)[:20]}"

    # 3. Load Model
    try:
        state_dict = torch.load(model_path, map_location=device)
        
        # Infer dims
        input_dims = {}
        # Check for .net.0.weight (MLPBlock) or .0.weight (Sequential)
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
            
        if not input_dims:
             return None, None, "Unknown Arch"

        # Check compatibility with data
        # If preprocessor produced different dims than model expects, it will fail at forward
        # We can check here
        if 'temporal' in input_dims and X_t.shape[1] != input_dims['temporal']:
             return None, None, f"Dim Mismatch: T {X_t.shape[1]} vs {input_dims['temporal']}"

        num_classes = 4
        model = CrashSeverityNet(input_dims, num_classes).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
    except Exception as e:
        return None, None, f"Load Error: {str(e)[:20]}"

    # 4. Inference
    try:
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
        # Recall for Class 4 (Index 3)
        # Calculate per-class recall manually or use recall_score with average=None
        recalls = recall_score(all_targets, all_preds, average=None, labels=[0,1,2,3], zero_division=0)
        sev4_recall = recalls[3] if len(recalls) > 3 else 0.0
        
        return acc, sev4_recall, "OK"
        
    except Exception as e:
        return None, None, f"Infer Error: {str(e)[:20]}"

def main():
    print("Scanning for models...")
    # Find all .pt files in models/
    model_files = [os.path.join("models", f) for f in os.listdir("models") if f.endswith(".pt")]
    
    print("Loading Test Data (Subset)...")
    if os.path.exists("US_Accidents_small.csv"):
        df = load_full_dataset("US_Accidents_small.csv")
    elif os.path.exists("data/raw/US_Accidents_March23.csv"):
        df = pd.read_csv("data/raw/US_Accidents_March23.csv", nrows=10000)
    else:
        print("No data found.")
        return

    df = clean_and_engineer_features(df)
    # Sample 2000 for testing
    if len(df) > 2000:
        df_test = df.sample(2000, random_state=42)
    else:
        df_test = df
        
    print(f"Evaluating {len(model_files)} models on {len(df_test)} samples...")
    with open("model_search_results.txt", "w") as log:
        log.write(f"{'File Path':<40} | {'Accuracy':<10} | {'Sev 4 Recall':<12} | {'Status':<20}\n")
        log.write("-" * 90 + "\n")
        
        for f in model_files:
            acc, sev4, status = evaluate_model(f, df_test)
            
            if acc is not None:
                acc_str = f"{acc:.4f}"
                sev4_str = f"{sev4:.4f}"
                
                # Candidate Criteria (Relaxed for logging)
                if acc > 0.50 and sev4 > 0.10:
                    status = "**CANDIDATE**"
                    candidates.append((f, acc, sev4))
            else:
                acc_str = "N/A"
                sev4_str = "N/A"
                
            line = f"{f:<40} | {acc_str:<10} | {sev4_str:<12} | {status:<20}\n"
            print(line.strip())
            log.write(line)
            
        log.write("-" * 90 + "\n")
        if candidates:
            best_model = max(candidates, key=lambda x: x[1] + x[2]) # Simple sum score
            log.write(f"Best Candidate: {best_model[0]} (Acc: {best_model[1]:.4f}, Sev4: {best_model[2]:.4f})\n")
        else:
            log.write("No candidates found meeting criteria (Acc > 0.50, Sev4 > 0.10).\n")

if __name__ == "__main__":
    main()
