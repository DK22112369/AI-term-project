import argparse
import os
import sys
import json
import torch
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.early_fusion_mlp import EarlyFusionMLP
from models.crash_severity_net import CrashSeverityNet
from data.preprocess_us_accidents import load_preprocessors, clean_and_engineer_features

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model file")
    parser.add_argument("--preprocessor_path", type=str, required=True, help="Path to preprocessors .joblib")
    parser.add_argument("--model_type", type=str, default="crash_severity_net", choices=["crash_severity_net", "early_mlp"])
    parser.add_argument("--input_json", type=str, help="JSON string or path to JSON file with input data")
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = "cpu"
    
    # 1. Load Input
    if os.path.exists(args.input_json):
        with open(args.input_json, 'r') as f:
            data = json.load(f)
    else:
        data = json.loads(args.input_json)
        
    # Convert to DataFrame (expecting list of dicts or single dict)
    if isinstance(data, dict):
        data = [data]
    df = pd.DataFrame(data)
    
    print("Input Data:")
    print(df)
    
    # 2. Preprocess
    # Note: clean_and_engineer_features expects certain columns. 
    # For inference, we assume input has raw fields.
    # We might need to handle missing columns if input is partial.
    df = clean_and_engineer_features(df)
    
    # Load Preprocessors
    bundle = load_preprocessors(args.preprocessor_path)
    
    # Transform
    # We duplicate the transform logic here slightly to avoid dependency on 'Severity' column which might not exist in inference
    # But transform_with_preprocessors expects Severity for y.
    # Let's manually transform X.
    
    X_driver = bundle["driver_preprocessor"].transform(df[bundle["driver_features"]])
    X_env = bundle["env_preprocessor"].transform(df[bundle["env_features"]])
    X_time = bundle["time_preprocessor"].transform(df[bundle["time_features"]])
    
    def to_dense(x): return x.toarray() if hasattr(x, 'toarray') else x
    X_driver = to_dense(X_driver)
    X_env = to_dense(X_env)
    X_time = to_dense(X_time)
    
    X_d_t = torch.tensor(X_driver.astype(np.float32)).to(device)
    X_e_t = torch.tensor(X_env.astype(np.float32)).to(device)
    X_t_t = torch.tensor(X_time.astype(np.float32)).to(device)
    
    # 3. Load Model
    # We need dimensions. 
    # Ideally these should be saved in config, but we can infer from preprocessors if we trust they match.
    # Or we can load the model state dict and check shapes (harder).
    # For this script, we assume the dimensions match the transformed data.
    d_in = X_d_t.shape[1]
    e_in = X_e_t.shape[1]
    t_in = X_t_t.shape[1]
    num_classes = 4
    
    if args.model_type == "crash_severity_net":
        model = CrashSeverityNet(d_in, e_in, t_in, num_classes).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            logits = model(X_d_t, X_e_t, X_t_t)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1) + 1 # 1-based Severity
            
    elif args.model_type == "early_mlp":
        input_dim = d_in + e_in + t_in
        model = EarlyFusionMLP(input_dim, num_classes).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            logits = model(X_d_t, X_e_t, X_t_t)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1) + 1
            
    # 4. Output
    results = []
    for i in range(len(preds)):
        results.append({
            "predicted_severity": int(preds[i]),
            "probabilities": probs[i].tolist()
        })
        
    print("\nInference Results:")
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
