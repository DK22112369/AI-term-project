import torch
import joblib
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_shapes():
    # 1. Check Preprocessors
    prep_path = "models/crash_severity_net_focal_time_preprocessors.joblib"
    if os.path.exists(prep_path):
        print(f"Loading {prep_path}...")
        bundle = joblib.load(prep_path)
        weather_prep = bundle.get("weather_preprocessor")
        if weather_prep:
            # Try to find OneHotEncoder
            # It might be a Pipeline or ColumnTransformer
            print("Weather Preprocessor:", type(weather_prep))
            try:
                # Assuming it's a pipeline or has 'transform'
                # We can't easily see output dim without data, but we can check fitted attributes
                if hasattr(weather_prep, 'transformers_'):
                    for name, trans, cols in weather_prep.transformers_:
                        print(f"  Transformer: {name}, Type: {type(trans)}")
                        if hasattr(trans, 'categories_'):
                            print(f"    Categories: {[len(c) for c in trans.categories_]}")
            except Exception as e:
                print(f"  Error inspecting preprocessor: {e}")
    else:
        print(f"File not found: {prep_path}")

    # 2. Check Model
    model_path = "models/crash_severity_net_focal_time.pt"
    if os.path.exists(model_path):
        print(f"Loading {model_path}...")
        state_dict = torch.load(model_path, map_location='cpu')
        if "encoders.weather.net.0.weight" in state_dict:
            print(f"  encoders.weather.net.0.weight shape: {state_dict['encoders.weather.net.0.weight'].shape}")
        else:
            print("  encoders.weather.net.0.weight not found in state_dict")
    else:
        print(f"File not found: {model_path}")

if __name__ == "__main__":
    check_shapes()
