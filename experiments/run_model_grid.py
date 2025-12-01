import os
import sys
import json
import argparse
import subprocess
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_single_experiment(exp_config, common_args):
    """
    Runs a single experiment based on configuration.
    Decides whether to call scripts/train.py or baselines/train_baseline_ml.py
    """
    # Merge common args with specific experiment args
    config = common_args.copy()
    config.update(exp_config)
    
    print(f"\n{'='*60}")
    print(f"Running Experiment: {config['id']}")
    print(f"Model: {config['model_type']}")
    print(f"{'='*60}")
    
    # Check if result already exists
    result_dir = "results"
    exp_id = config["id"]
    result_file = os.path.join(result_dir, f"{exp_id}.json")
    if os.path.exists(result_file):
        print(f"Skipping {exp_id}, result already exists.")
        return True

    # Determine which script to run
    ml_models = ["rf", "xgb", "catboost", "lgbm"]
    dl_models = ["crash_severity_net", "early_mlp", "tab_transformer"]
    
    cmd = []
    if config['model_type'] in ml_models:
        script = "baselines/train_baseline_ml.py"
        cmd = [
            sys.executable, script,
            "--model_type", config["model_type"],
            "--split_strategy", config.get("split_strategy", "time"),
            "--sample_frac", str(config.get("sample_frac", 0.1)),
            "--seed", str(config.get("seed", 42)),
            "--exp_id", config["id"]
        ]
    elif config['model_type'] in dl_models:
        script = "scripts/train.py"
        cmd = [
            sys.executable, script,
            "--model_type", config["model_type"],
            "--loss_type", config.get("loss_type", "ce"),
            "--split_strategy", config.get("split_strategy", "time"),
            "--epochs", str(config.get("epochs", 20)),
            "--sample_frac", str(config.get("sample_frac", 0.1)),
            "--seed", str(config.get("seed", 42)),
            "--exp_id", config["id"]
        ]
        if config.get("use_sampler"):
            cmd.append("--use_sampler")
        if config.get("use_smote"):
            cmd.append("--use_smote")
        if config.get("gamma"):
            cmd.extend(["--gamma", str(config.get("gamma"))])
        
        # Region Generalization Args
        if config.get("region_a_states"):
            cmd.append("--region_a_states")
            cmd.extend(config["region_a_states"])
        if config.get("region_b_states"):
            cmd.append("--region_b_states")
            cmd.extend(config["region_b_states"])
    else:
        print(f"Unknown model type: {config['model_type']}")
        return False

    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"Experiment {config['id']} completed in {duration:.2f} seconds.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment {config['id']}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Experiment Grid from Config")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return

    with open(args.config, 'r') as f:
        grid_config = json.load(f)
        
    print(f"Loaded Experiment Grid: {grid_config.get('description', 'No description')}")
    common_args = grid_config.get("common_args", {})
    experiments = grid_config.get("experiments", [])
    
    print(f"Total experiments: {len(experiments)}")
    
    successful = 0
    failed = 0
    
    for i, exp in enumerate(experiments):
        print(f"\nProgress: [{i+1}/{len(experiments)}]")
        if run_single_experiment(exp, common_args):
            successful += 1
        else:
            failed += 1
            
    print(f"\n{'='*60}")
    print(f"Grid Execution Complete.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
