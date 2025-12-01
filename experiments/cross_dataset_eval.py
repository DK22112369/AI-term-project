import argparse
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crash_severity_net import CrashSeverityNet, infer_input_dims
from utils.metrics import evaluate_metrics, save_metrics

def evaluate_cross_dataset(model_path, data_path, output_dir, device="cpu"):
    """
    Evaluates a trained model on preprocessed cross-dataset data (e.g., FARS).
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Infer dimensions
    input_dims = infer_input_dims(checkpoint)
    num_classes = 4 # Fixed for this project
    
    # Initialize Model
    model = CrashSeverityNet(input_dims, num_classes).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    X_t = torch.FloatTensor(data['X_t']).to(device)
    X_w = torch.FloatTensor(data['X_w']).to(device)
    X_r = torch.FloatTensor(data['X_r']).to(device)
    X_s = torch.FloatTensor(data['X_s']).to(device)
    y = torch.LongTensor(data['y']).to(device)
    
    dataset = TensorDataset(X_t, X_w, X_r, X_s, y)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    print("Running inference...")
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for xt, xw, xr, xs, target in loader:
            inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
            out = model(inputs)
            
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    # Calculate Metrics
    print("Calculating metrics...")
    metrics = evaluate_metrics(all_targets, all_preds, y_probs=all_probs)
    
    # Save Results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    save_path = os.path.join(output_dir, "cross_dataset_metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Results saved to {save_path}")
    
    # Print Key Metrics
    print("\n=== Cross-Dataset Evaluation Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    # Fatal Class (Index 3)
    fatal_metrics = metrics['classification_report'].get('3', {})
    print(f"Fatal Recall: {fatal_metrics.get('recall', 0):.4f}")
    print(f"Fatal Precision: {fatal_metrics.get('precision', 0):.4f}")
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Dataset Evaluation (US Accidents -> FARS)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to preprocessed .npz data")
    parser.add_argument("--output_dir", type=str, default="results/cross_dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    evaluate_cross_dataset(args.model_path, args.data_path, args.output_dir, args.device)
