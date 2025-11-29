import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Add parent directory to path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess_us_accidents import (
    load_full_dataset, 
    stratified_sample, 
    clean_and_engineer_features, 
    fit_feature_transformers,
    transform_with_preprocessors,
    time_based_split,
    save_preprocessors
)
from utils.metrics import evaluate_metrics, save_metrics
from visualization.plots import plot_confusion_matrix
from utils.common import set_seed

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Baseline ML Models (RF, XGB, CatBoost, LGBM)")
    parser.add_argument("--data_path", type=str, default="data/raw/US_Accidents_March23.csv")
    parser.add_argument("--sample_frac", type=float, default=0.1)
    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "xgb", "catboost", "lgbm"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split_strategy", type=str, default="random", choices=["random", "time"])
    return parser.parse_args()

def main():
    args = parse_arguments()
    set_seed(args.seed)
    print(f"Training Baseline Model: {args.model_type.upper()}, Split: {args.split_strategy}")
    
    # 1. Load & Preprocess Data
    if not os.path.exists(args.data_path):
        if os.path.exists("US_Accidents_small.csv"):
            args.data_path = "US_Accidents_small.csv"
        elif os.path.exists("US_Accidents_March23.csv"):
             args.data_path = "US_Accidents_March23.csv"
        else:
            print(f"Error: Dataset not found at {args.data_path}")
            return

    df = load_full_dataset(args.data_path)
    df = stratified_sample(df, frac=args.sample_frac)
    df = clean_and_engineer_features(df)
    
    # 2. Split Data
    if args.split_strategy == "random":
        print("Splitting data (Stratified Random)...")
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df["Severity"])
    else:
        d_tr, d_val, d_te = time_based_split(df)
        df_train = pd.concat([d_tr, d_val])
        df_test = d_te
    
    # 3. Fit & Transform
    bundle = fit_feature_transformers(df_train)
    
    # Save Preprocessors
    experiment_name = f"baseline_{args.model_type}"
    if args.split_strategy == "time": experiment_name += "_time"
    save_preprocessors(bundle, f"models/{experiment_name}_preprocessors.joblib")
    
    print("Transforming datasets...")
    # Note: We use CPU for sklearn models
    X_d_train, X_e_train, X_t_train, y_train = transform_with_preprocessors(df_train, bundle, device='cpu')
    X_d_test,  X_e_test,  X_t_test,  y_test  = transform_with_preprocessors(df_test,  bundle, device='cpu')
    
    # Concatenate features for ML models
    X_train = np.concatenate([X_d_train.numpy(), X_e_train.numpy(), X_t_train.numpy()], axis=1)
    X_test = np.concatenate([X_d_test.numpy(), X_e_test.numpy(), X_t_test.numpy()], axis=1)
    y_train = y_train.numpy()
    y_test = y_test.numpy()
    
    # 4. Train Model
    if args.model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
    elif args.model_type == "xgb":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1, eval_metric='mlogloss')
        except ImportError:
            print("Error: XGBoost not installed. Please install it or use 'rf'.")
            return
    elif args.model_type == "catboost":
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(iterations=100, random_seed=args.seed, verbose=0, allow_writing_files=False)
        except ImportError:
            print("Error: CatBoost not installed.")
            return
    elif args.model_type == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1, verbose=-1)
        except ImportError:
            print("Error: LightGBM not installed.")
            return
            
    print(f"Training {args.model_type.upper()}...")
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    metrics = evaluate_metrics(y_test, y_pred)
    print(metrics['classification_report'])
    
    # 6. Save Results & Config
    config_path = f"results/{experiment_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    save_metrics(metrics, vars(args), f"results/{experiment_name}.json")
    
    plot_confusion_matrix(metrics['confusion_matrix'], classes=["1", "2", "3", "4"],
                          title=f"Confusion Matrix ({args.model_type.upper()})",
                          save_path=f"results/confmat_{experiment_name}.png")
                          
    # 7. Feature Importance
    print("Extracting Feature Importance...")
    try:
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'importance': importances})
        imp_df.to_csv(f"results/{experiment_name}_feature_importance.csv", index=True)
        print(f"Feature importance saved to results/{experiment_name}_feature_importance.csv")
    except Exception as e:
        print(f"Could not extract feature importance: {e}")

    # Save Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{experiment_name}.joblib")
    print(f"Model saved to models/{experiment_name}.joblib")

if __name__ == "__main__":
    main()
