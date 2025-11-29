import pandas as pd
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

FEATURE_PATH = Path("data/processed/us_accidents_features.csv")
MODEL_DIR = Path("models")
BEST_MODEL_PATH = MODEL_DIR / "severity_best.joblib"

def load_data():
    print(f"Loading features from {FEATURE_PATH}...")
    # TODO: Load data
    # TODO: Separate X and y (Target: "Severity")
    pass

def train_models(X_train, y_train):
    print("Training models...")
    # TODO: Train LogisticRegression
    # TODO: Train RandomForestClassifier
    # TODO: Train MLPClassifier
    # Return a dictionary of trained models
    pass

def evaluate_models(models, X_val, y_val):
    print("Evaluating models...")
    # TODO: Predict on validation set
    # TODO: Calculate Accuracy, Macro F1
    # TODO: Print Classification Report
    # Return best model based on F1 score
    pass

def save_best_model(model, path):
    print(f"Saving best model to {path}...")
    # TODO: Save model using joblib
    pass

def main():
    if not FEATURE_PATH.exists():
        print(f"Error: {FEATURE_PATH} not found. Run feature_engineering.py first.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: Orchestrate the flow
    # X, y = load_data()
    # X_train, X_val, X_test, y_train, y_val, y_test = ... (split)
    
    # models = train_models(X_train, y_train)
    # best_model = evaluate_models(models, X_val, y_val)
    # save_best_model(best_model, BEST_MODEL_PATH)
    
    print("Done.")

if __name__ == "__main__":
    main()
