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
    
    def transform_group(group_name):
        preprocessor = bundle.get(f"{group_name}_preprocessor")
        features = bundle.get(f"{group_name}_features")
        if preprocessor and features:
            # Ensure columns exist
            for col in features:
                if col not in df.columns:
                    df[col] = 0 # Default value
            X = preprocessor.transform(df[features])
            return X.toarray() if hasattr(X, 'toarray') else X
        return np.zeros((len(df), 0))

    X_temporal = transform_group("temporal")
    X_weather = transform_group("weather")
    X_road = transform_group("road")
    X_spatial = transform_group("spatial")
    
    X_t_t = torch.tensor(X_temporal.astype(np.float32)).to(device)
    X_w_t = torch.tensor(X_weather.astype(np.float32)).to(device)
    X_r_t = torch.tensor(X_road.astype(np.float32)).to(device)
    X_s_t = torch.tensor(X_spatial.astype(np.float32)).to(device)
    
    # 3. Load Model
    # We need dimensions. 
    t_in = X_t_t.shape[1]
    w_in = X_w_t.shape[1]
    r_in = X_r_t.shape[1]
    s_in = X_s_t.shape[1]
    num_classes = 4
    
    if args.model_type == "crash_severity_net":
        input_dims = {
            'temporal': t_in,
            'weather': w_in,
            'road': r_in,
            'spatial': s_in
        }
        model = CrashSeverityNet(input_dims, num_classes).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            inputs = {'temporal': X_t_t, 'weather': X_w_t, 'road': X_r_t, 'spatial': X_s_t}
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1) + 1 # 1-based Severity
            
    elif args.model_type == "early_mlp":
        input_dim = t_in + w_in + r_in + s_in
        model = EarlyFusionMLP(input_dim, num_classes).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # Concatenate
            concat_input = torch.cat([X_t_t, X_w_t, X_r_t, X_s_t], dim=1)
            logits = model(concat_input)
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
