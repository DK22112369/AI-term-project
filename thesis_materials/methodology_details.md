# Methodology Details

## 1. Model Architecture: CrashSeverityNet (Group-wise Late Fusion)

The proposed model, **CrashSeverityNet**, utilizes a Group-wise Late Fusion architecture designed to process heterogeneous data sources independently before fusion.

### 1.1 Feature Groups
The input features are divided into four distinct groups based on their semantic meaning:

1.  **Temporal Group**: Captures time-related patterns.
    - Features: `Start_Hour`, `Start_DayOfWeek`, `Start_Month`, `Duration_minutes`
2.  **Weather Group**: Captures environmental conditions.
    - Features: `Temperature(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`, `Weather_Condition`
3.  **Road Group**: Captures infrastructure and road layout.
    - Features: `Bump`, `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Roundabout`, `Station`, `Stop`, `Traffic_Calming`, `Traffic_Signal`, `Turning_Loop`
4.  **Spatial Group**: Captures geographical location.
    - Features: `Start_Lat`, `Start_Lng`, `Distance(mi)`

### 1.2 Sub-Network Architecture
Each feature group is processed by an independent Multi-Layer Perceptron (MLP) block with the following structure:
- **Input Layer**: Dimension corresponds to the number of features in the group (after One-Hot Encoding).
- **Hidden Layer 1**: Linear(Input -> 64) -> ReLU -> BatchNorm1d -> Dropout(0.3)
- **Hidden Layer 2**: Linear(64 -> 64) -> ReLU -> BatchNorm1d -> Dropout(0.3)

### 1.3 Fusion Layer
The outputs of the four sub-networks are concatenated to form a latent representation vector.
- **Concatenation**: 64 (Temporal) + 64 (Weather) + 64 (Road) + 64 (Spatial) = 256 dimensions.
- **Fusion MLP**:
    - Linear(256 -> 128) -> ReLU -> BatchNorm1d -> Dropout(0.3)
    - Linear(128 -> 4) (Output Layer for 4 Severity Classes)

## 2. Training Configuration

The best performing model was trained using the following hyperparameters:

- **Loss Function**: **Focal Loss** (to address class imbalance)
    - Gamma ($\gamma$): 2.0
    - Alpha ($\alpha$): Inverse Class Frequencies (Balanced)
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 128
- **Epochs**: 20 (with Early Stopping)
- **Early Stopping Patience**: 5 epochs (monitoring Validation Loss)
- **Data Splitting**: Time-based Splitting (Train: 70%, Val: 10%, Test: 20%) to prevent data leakage.

## 3. Implementation Stack
- **Framework**: PyTorch
- **Hardware**: CUDA (GPU Accelerated)
- **Reproducibility**: Random Seed fixed to 42.
