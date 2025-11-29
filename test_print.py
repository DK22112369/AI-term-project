print("Hello from Python")
import os
print(f"CWD: {os.getcwd()}")
try:
    import matplotlib.pyplot as plt
    print("Matplotlib imported")
except ImportError as e:
    print(f"Matplotlib import failed: {e}")
