# 2. Methodology Specifications

## 2.1 Experimental Environment

| Component | Specification |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Framework** | PyTorch 2.0+ |
| **Hardware** | NVIDIA GPU (CUDA 11.8) |
| **OS** | Windows 10/11 |

## 2.2 Network Architecture (CrashSeverityNet)

The proposed model utilizes a **Group-wise Late Fusion** architecture to process heterogeneous data sources effectively.

### Input Dimensions
| Feature Group | Input Dimension | Description |
| :--- | :--- | :--- |
| **Temporal** | 14 | Hour, Day, Month, Duration (Cyclical encoded) |
| **Weather** | 130 | Temp, Humidity, Pressure, Condition (One-Hot) |
| **Road** | 12 | Traffic Signal, Junction, Crossing, etc. (Binary) |
| **Spatial** | 3 | Lat, Lng, Distance |

### Fusion Layer
- **Encoder Output:** Each group is processed by a dedicated MLP encoder outputting a **64-dimensional** embedding.
- **Fusion Input:** Concatenation of 4 embeddings ($64 \times 4 = 256$).
- **Fusion MLP:** $256 \to 128 \to 64 \to 4$ (Output Classes).

## 2.3 Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | Weight Decay = 1e-4 |
| **Learning Rate** | 1e-3 | With Cosine Annealing Scheduler |
| **Batch Size** | 256 | - |
| **Loss Function** | Weighted Cross Entropy | Weights inverse to class frequency |
| **Epochs** | 10-20 | With Early Stopping (Patience=5) |
