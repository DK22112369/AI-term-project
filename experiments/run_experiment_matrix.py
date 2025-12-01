import os
import sys
import subprocess
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_experiment(config):
    """
    Runs a single experiment using scripts/train.py
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment: {config['name']}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "scripts/train.py",
        "--model_type", config["model_type"],
        "--loss_type", config["loss_type"],
        "--split_strategy", config["split_strategy"],
        "--epochs", str(config.get("epochs", 20)),
        "--sample_frac", str(config.get("sample_frac", 0.1)),
        "--seed", str(config.get("seed", 42))
    ]
    
    if config.get("use_sampler"):
        cmd.append("--use_sampler")
        
    if config.get("use_smote"):
        cmd.append("--use_smote")
        
    if "gamma" in config:
        cmd.extend(["--gamma", str(config["gamma"])])

    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        # Run the command and wait for it to complete
        result = subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"Experiment {config['name']} completed in {duration:.2f} seconds.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment {config['name']}: {e}")
        return False

def main():
    # Define the Experiment Matrix
    # This list covers the key ablation studies for the thesis/journal
    EXPERIMENTS = [
        # 1. Main Proposal (Fail-Safe)
        {
            "name": "crash_latefusion_weighted_time",
            "model_type": "crash_severity_net",
            "loss_type": "ce_weighted",
            "split_strategy": "time",
            "use_sampler": False,
        },
        # 2. Ablation: Loss Function (Standard CE)
        {
            "name": "crash_latefusion_ce_time",
            "model_type": "crash_severity_net",
            "loss_type": "ce",
            "split_strategy": "time",
            "use_sampler": False,
        },
        # 3. Ablation: Loss Function (Focal Loss)
        {
            "name": "crash_latefusion_focal_time",
            "model_type": "crash_severity_net",
            "loss_type": "focal",
            "split_strategy": "time",
            "use_sampler": False,
            "gamma": 2.0
        },
        # 4. Ablation: Architecture (Early Fusion Baseline)
        {
            "name": "crash_earlyfusion_weighted_time",
            "model_type": "early_mlp",
            "loss_type": "ce_weighted",
            "split_strategy": "time",
            "use_sampler": False,
        },
        # 5. Ablation: Sampling Strategy (Oversampling instead of Weighted Loss)
        {
            "name": "crash_latefusion_sampler_time",
            "model_type": "crash_severity_net",
            "loss_type": "ce", # Standard CE with Sampler
            "split_strategy": "time",
            "use_sampler": True,
        }
    ]

    print(f"Starting Experiment Matrix. Total experiments: {len(EXPERIMENTS)}")
    
    successful = 0
    failed = 0
    
    for i, exp_config in enumerate(EXPERIMENTS):
        print(f"\nProgress: [{i+1}/{len(EXPERIMENTS)}]")
        if run_experiment(exp_config):
            successful += 1
        else:
            failed += 1
            
    print(f"\n{'='*60}")
    print(f"Matrix Execution Complete.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Check 'results/' directory for logs and artifacts.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
