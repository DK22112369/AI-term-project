
import argparse
import os
import sys
import numpy as np
import pandas as pd
import shap
import torch
import matplotlib.pyplot as plt
import joblib
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crash_severity_net import CrashSeverityNet, infer_input_dims
from data.preprocess_us_accidents import transform_with_preprocessors, clean_and_engineer_features

def parse_arguments():
    parser = argparse.ArgumentParser(description="SHAP Analysis for CrashSeverityNet")
    parser.add_argument("--model_path", type=str, default="models/crash_severity_net_ce_weighted_time.pt")
    parser.add_argument("--preprocessor_path", type=str, default="models/crash_severity_net_ce_weighted_time_preprocessors.joblib")
    parser.add_argument("--data_path", type=str, default="US_Accidents_small.csv")
    parser.add_argument("--output_dir", type=str, default="results/shap")
    parser.add_argument("--target_severity", type=int, default=4, choices=[1, 2, 3, 4], help="Severity class to explain (1-4)")
    parser.add_argument("--n_background", type=int, default=50, help="Number of background samples")
    parser.add_argument("--n_test", type=int, default=20, help="Number of test samples to explain")
    parser.add_argument("--top_n", type=int, default=100, help="Number of top samples for scenario analysis")
    parser.add_argument("--scenario_summary", action="store_true", help="Generate scenario-level summary")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to load for scenario analysis")
    return parser.parse_args()

def load_model_and_data(model_path, preprocessor_path, data_sample_path, device='cpu'):
    """
    Loads the trained model, preprocessors, and a sample of data for SHAP analysis.
    """
    print(f"Loading preprocessors from {preprocessor_path}...")
    bundle = joblib.load(preprocessor_path)
    
    print(f"Loading data sample from {data_sample_path}...")
    # Load small sample for background (e.g., 100 samples)
    try:
        df = pd.read_csv(data_sample_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decode error, trying latin1...")
        df = pd.read_csv(data_sample_path, encoding='latin1')
    
    # Engineer features
    df = clean_and_engineer_features(df)
        
    if len(df) > 500:
        df = df.sample(500, random_state=42)
    
    # Transform data
    X_t, X_w, X_r, X_s, y = transform_with_preprocessors(df, bundle, device=device)
    
    return X_t, X_w, X_r, X_s, bundle, df

def shap_analysis(model_path, preprocessor_path, data_path, output_dir="results/shap", target_severity=4, n_bg=50, n_test=20):
    """
    Runs SHAP analysis with advanced features.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu" # SHAP often works better on CPU for compatibility
    
    # 1. Load Data
    X_t, X_w, X_r, X_s, bundle, df_raw = load_model_and_data(model_path, preprocessor_path, data_path, device)
    
    # 2. Load Model
    print("Initializing model...")
    checkpoint = torch.load(model_path, map_location=device)
    input_dims = infer_input_dims(checkpoint)
    model = CrashSeverityNet(input_dims, num_classes=4).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3. Wrapper for SHAP (KernelExplainer)
    X_bg = torch.cat([X_t, X_w, X_r, X_s], dim=1).detach().cpu().numpy()
    
    t_dim = X_t.shape[1]
    w_dim = X_w.shape[1]
    r_dim = X_r.shape[1]
    s_dim = X_s.shape[1]
    
    def model_predict(x_numpy):
        # x_numpy: (batch, total_features)
        x_tensor = torch.FloatTensor(x_numpy).to(device)
        
        # Split back
        xt = x_tensor[:, :t_dim]
        xw = x_tensor[:, t_dim:t_dim+w_dim]
        xr = x_tensor[:, t_dim+w_dim:t_dim+w_dim+r_dim]
        xs = x_tensor[:, t_dim+w_dim+r_dim:]
        
        inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # 4. Calculate SHAP
    print(f"Calculating SHAP values for Target Severity {target_severity}...")
    # Use raw background instead of kmeans to avoid issues
    background_summary = X_bg[:n_bg]
        
    explainer = shap.KernelExplainer(model_predict, background_summary)
    
    # Explain test set
    X_test_shap = X_bg[:n_test] 
    shap_values = explainer.shap_values(X_test_shap)

    # Feature names need to be reconstructed
    feature_names = []
    feature_names.extend([f"Temporal_{i}" for i in range(t_dim)])
    feature_names.extend([f"Weather_{i}" for i in range(w_dim)])
    feature_names.extend([f"Road_{i}" for i in range(r_dim)])
    feature_names.extend([f"Spatial_{i}" for i in range(s_dim)])

    # Debug Prints to File
    with open("shap_shapes.txt", "w") as f:
        f.write(f"X_test_shap shape: {X_test_shap.shape}\n")
        f.write(f"Feature names length: {len(feature_names)}\n")
        if isinstance(shap_values, list):
            f.write(f"shap_values is a list of length {len(shap_values)}\n")
            f.write(f"shap_values[0] shape: {shap_values[0].shape}\n")
        else:
            f.write(f"shap_values shape: {shap_values.shape}\n")

    # 5. Plot & Save Summary
    print("Saving SHAP summary plot...")
    
    # Target index (0-indexed) -> Severity 1 is index 0
    target_idx = target_severity - 1
    
    # Handle different shap_values formats
    if isinstance(shap_values, list):
        shap_values_target = shap_values[target_idx]
    elif len(shap_values.shape) == 3:
        # (n_samples, n_features, n_classes)
        shap_values_target = shap_values[:, :, target_idx]
    else:
        shap_values_target = shap_values

    plt.figure()
    # shap_values is a list of arrays (one for each class)
    shap.summary_plot(shap_values_target, X_test_shap, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary for Severity {target_severity}")
    plt.savefig(f"{output_dir}/shap_summary_sev{target_severity}.png", bbox_inches='tight')
    plt.close()
    
    # 6. Group Importance
    print("Calculating Group Importance...")
    # shap_values_target shape: (n_test, total_features)
    vals = np.abs(shap_values_target)
    
    group_imp = {
        "Temporal": float(np.sum(vals[:, :t_dim])),
        "Weather": float(np.sum(vals[:, t_dim:t_dim+w_dim])),
        "Road": float(np.sum(vals[:, t_dim+w_dim:t_dim+w_dim+r_dim])),
        "Spatial": float(np.sum(vals[:, t_dim+w_dim+r_dim:]))
    }
    
    # Print and Save Group Importance
    print("Group Importance:")
    print(json.dumps(group_imp, indent=4))
    
    with open(f"{output_dir}/group_importance_sev{target_severity}.json", "w") as f:
        json.dump(group_imp, f, indent=4)
        
    # Also save feature importance CSV
    # Mean absolute SHAP value for each feature
    mean_abs_shap = np.mean(np.abs(shap_values_target), axis=0)
    df_imp = pd.DataFrame({
        "feature_name": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    
    df_imp.to_csv(f"{output_dir}/shap_feature_importance_sev{target_severity}.csv", index=False)
    print(f"Feature importance saved to {output_dir}/shap_feature_importance_sev{target_severity}.csv")
    
    # Normalize
    total_imp = sum(group_imp.values())
    for k in group_imp:
        group_imp[k] /= total_imp
        
    # Save Group Importance
    df_imp = pd.DataFrame(list(group_imp.items()), columns=["Group", "Importance"])
    df_imp.to_csv(f"{output_dir}/group_importance_sev{target_severity}.csv", index=False)
    print(f"Group importance saved to {output_dir}/group_importance_sev{target_severity}.csv")
    
    print(f"SHAP analysis complete. Results saved to {output_dir}")

    # 7. Scenario Analysis (Optional)
    if args.scenario_summary:
        run_scenario_analysis(model, X_bg, df_raw, feature_names, target_severity, args.top_n, output_dir, device, t_dim, w_dim, r_dim, s_dim)

def run_scenario_analysis(model, X_bg, df_raw, feature_names, target_severity, top_n, output_dir, device, t_dim, w_dim, r_dim, s_dim):
    print(f"\nRunning Scenario Analysis for Top {top_n} Fatal Cases...")
    
    # 1. Predict on all data (or a large subset)
    # We need to map X_bg back to df_raw indices if possible, or just use X_bg as the source
    # For simplicity, we assume X_bg corresponds to the processed version of df_raw (which it does in load_model_and_data)
    
    # Predict
    model.eval()
    all_probs = []
    batch_size = 1024
    
    X_tensor = torch.FloatTensor(X_bg).to(device)
    n_samples = len(X_tensor)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = X_tensor[i:i+batch_size]
            
            xt = batch[:, :t_dim]
            xw = batch[:, t_dim:t_dim+w_dim]
            xr = batch[:, t_dim+w_dim:t_dim+w_dim+r_dim]
            xs = batch[:, t_dim+w_dim+r_dim:]
            
            inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
            out = model(inputs)
            probs = torch.softmax(out, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
    all_probs = np.array(all_probs)
    
    # 2. Filter for Target Severity
    # We want cases where the model predicts the target severity with high probability
    target_idx = target_severity - 1
    target_probs = all_probs[:, target_idx]
    
    # Sort by probability descending
    top_indices = np.argsort(target_probs)[::-1][:top_n]
    
    print(f"Found {len(top_indices)} top samples.")
    
    # 3. Extract Scenario Data
    # We need to map back to original features if possible. 
    # Since df_raw is available, we can use iloc.
    
    top_cases = df_raw.iloc[top_indices].copy()
    top_cases['Model_Prob'] = target_probs[top_indices]
    
    # Save Top Cases
    os.makedirs(f"{output_dir}_fatal", exist_ok=True)
    save_dir = f"{output_dir}_fatal"
    
    top_cases.to_csv(f"{save_dir}/top_fatal_cases.csv", index=False)
    print(f"Saved top cases to {save_dir}/top_fatal_cases.csv")
    
    # 4. Generate Summary
    summary = {
        "Total_Samples": len(top_cases),
        "Avg_Prob": float(top_cases['Model_Prob'].mean()),
        "Common_Weather": top_cases['Weather_Condition'].mode()[0] if 'Weather_Condition' in top_cases else "N/A",
        "Common_Hour": int(top_cases['Start_Hour'].mode()[0]) if 'Start_Hour' in top_cases else -1,
        "High_Risk_Road_Features": []
    }
    
    # Check Road Features
    road_feats = ['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
    for rf in road_feats:
        if rf in top_cases.columns:
            freq = top_cases[rf].mean()
            if freq > 0.1: # If present in > 10% of top fatal cases
                summary["High_Risk_Road_Features"].append(f"{rf} ({freq*100:.1f}%)")
                
    # Save Summary CSV
    pd.DataFrame([summary]).to_csv(f"{save_dir}/scenario_summary.csv", index=False)
    
    # 5. Write Markdown Report
    md_path = "docs/shap_scenario_analysis.md"
    with open(md_path, "w") as f:
        f.write(f"# SHAP Scenario Analysis: Top {top_n} Fatal Cases\n\n")
        f.write(f"**Target Severity**: {target_severity}\n")
        f.write(f"**Average Model Confidence**: {summary['Avg_Prob']:.4f}\n\n")
        
        f.write("## Common Patterns\n")
        f.write(f"- **Weather**: {summary['Common_Weather']}\n")
        f.write(f"- **Time of Day**: {summary['Common_Hour']}:00\n")
        f.write(f"- **High Risk Infrastructure**: {', '.join(summary['High_Risk_Road_Features'])}\n\n")
        
        f.write("## Top 5 Most Dangerous Cases\n")
        f.write("| Probability | Weather | Hour | Road Features |\n")
        f.write("|---|---|---|---|\n")
        
        for i in range(min(5, len(top_cases))):
            row = top_cases.iloc[i]
            # Collect active road features
            active_rf = [rf for rf in road_feats if rf in row and row[rf] == 1]
            rf_str = ", ".join(active_rf) if active_rf else "None"
            
            weather = row.get('Weather_Condition', 'N/A')
            hour = row.get('Start_Hour', 'N/A')
            prob = row['Model_Prob']
            
            f.write(f"| {prob:.4f} | {weather} | {hour} | {rf_str} |\n")
            
    print(f"Markdown report saved to {md_path}")

if __name__ == "__main__":
    args = parse_arguments()
    
    if os.path.exists(args.model_path) and os.path.exists(args.preprocessor_path):
        shap_analysis(args.model_path, args.preprocessor_path, args.data_path, args.output_dir, 
                      args.target_severity, args.n_background, args.n_test)
    else:
        print(f"Model or preprocessor not found. Please run training first.")
