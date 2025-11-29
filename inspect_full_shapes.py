import torch
import sys

def inspect():
    try:
        sd = torch.load('models/crash_severity_net_ce_weighted.pt', map_location='cpu')
        with open('model_shapes.txt', 'w') as f:
            for k, v in sd.items():
                f.write(f"{k}: {v.shape}\n")
        print("Shapes written to model_shapes.txt")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
