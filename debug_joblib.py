import joblib
import sys
import os

path = "models/crash_severity_net_ce_weighted_time_preprocessors.joblib"
print(f"Attempting to load {path}")
print(f"File size: {os.path.getsize(path)} bytes")

try:
    bundle = joblib.load(path)
    print("Success!")
    print(f"Keys: {bundle.keys()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
