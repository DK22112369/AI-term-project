import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import torch
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. Feature Groups Definition
# ==========================================

# Group 1: Temporal Features
TEMPORAL_FEATURES = [
    'Start_Hour', 'Start_DayOfWeek', 'Start_Month', 'Duration_minutes'
]

# Group 2: Weather Features
WEATHER_FEATURES = [
    'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
    'Weather_Condition' # Categorical
]

# Group 3: Road Features (Infrastructure)
ROAD_FEATURES = [
    'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 
    'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'
]

# Group 4: Spatial Features
SPATIAL_FEATURES = [
    'Start_Lat', 'Start_Lng', 'Distance(mi)'
]

ALL_FEATURE_GROUPS = {
    'temporal': TEMPORAL_FEATURES,
    'weather': WEATHER_FEATURES,
    'road': ROAD_FEATURES,
    'spatial': SPATIAL_FEATURES
}

# ==========================================
# 2. Data Loading & Cleaning
# ==========================================

def load_full_dataset(path: str) -> pd.DataFrame:
    """Loads the dataset from CSV."""
    print(f"Loading dataset from {path}...")
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        print("UnicodeDecodeError detected. Retrying with encoding_errors='replace'...")
        df = pd.read_csv(path, encoding_errors='replace')
    return df

def stratified_sample(df: pd.DataFrame, frac: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Returns a stratified sample of the dataframe."""
    print(f"Sampling {frac*100}% of data (Stratified)...")
    return df.groupby('Severity', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=seed))

def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic cleaning and feature engineering.
    """
    print("Cleaning and engineering features...")
    df = df.copy()
    
    # 1. Handle Missing Values (Basic)
    # Drop rows with missing target
    df = df.dropna(subset=['Severity'])
    
    # Fill numerical missing with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Fill categorical missing with mode or 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
    # 2. Feature Engineering
    
    # Convert Start_Time and End_Time to datetime
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    
    # Drop rows with invalid Start_Time
    df = df.dropna(subset=['Start_Time'])
    
    # Extract Temporal Features
    df['Start_Hour'] = df['Start_Time'].dt.hour
    df['Start_DayOfWeek'] = df['Start_Time'].dt.dayofweek
    df['Start_Month'] = df['Start_Time'].dt.month
    
    # Calculate Duration
    df['Duration_minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60.0
    df['Duration_minutes'] = df['Duration_minutes'].fillna(0)
    
    # Clean Boolean/Road features (ensure they are 0/1 or False/True)
    for col in ROAD_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(int)
            
    # Simplify Weather Condition (Optional)
    # OneHotEncoder will handle unknown categories automatically.
    
    return df

def time_based_split(df: pd.DataFrame, train_frac=0.6, val_frac=0.2):
    """
    Splits data based on time.
    """
    print("Sorting data by time for splitting...")
    df = df.sort_values('Start_Time')
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, val, test

# ==========================================
# 3. Preprocessing Pipelines (Transformers)
# ==========================================

def fit_feature_transformers(df_train: pd.DataFrame) -> dict:
    """
    Fits Scalers and Encoders on the training data for each group.
    Returns a dictionary of fitted transformers.
    """
    print("Fitting feature transformers...")
    
    transformers = {}
    
    for group_name, features in ALL_FEATURE_GROUPS.items():
        # Identify numerical and categorical features in this group
        # We check if they exist in df_train to avoid errors
        valid_features = [f for f in features if f in df_train.columns]
        
        num_cols = [f for f in valid_features if pd.api.types.is_numeric_dtype(df_train[f])]
        cat_cols = [f for f in valid_features if not pd.api.types.is_numeric_dtype(df_train[f])]
        
        # Build Transformer
        steps = []
        if num_cols:
            steps.append(('num', StandardScaler(), num_cols))
        if cat_cols:
            steps.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
            
        if steps:
            ct = ColumnTransformer(steps)
            ct.fit(df_train)
            transformers[f"{group_name}_preprocessor"] = ct
            transformers[f"{group_name}_features"] = valid_features
            
    return transformers

def transform_with_preprocessors(df: pd.DataFrame, bundle: dict, device: str = "cpu"):
    """
    Transforms the dataframe using the fitted bundle.
    Returns 4 tensors (Temporal, Weather, Road, Spatial) and Targets.
    """
    outputs = {}
    
    for group_name in ['temporal', 'weather', 'road', 'spatial']:
        preprocessor = bundle.get(f"{group_name}_preprocessor")
        if preprocessor:
            # Transform
            X = preprocessor.transform(df)
            # Convert to dense if sparse
            if hasattr(X, 'toarray'):
                X = X.toarray()
            outputs[group_name] = torch.tensor(X.astype(np.float32)).to(device)
        else:
            # Handle empty group if necessary
            outputs[group_name] = torch.tensor([]).to(device)
            
    y = df["Severity"].values - 1 # 0-indexed (0-3)
    y_t = torch.tensor(y).long().to(device)
    
    return outputs['temporal'], outputs['weather'], outputs['road'], outputs['spatial'], y_t

def save_preprocessors(bundle: dict, path: str):
    joblib.dump(bundle, path)
    print(f"Preprocessors saved to {path}")

def load_preprocessors(path: str) -> dict:
    return joblib.load(path)

# ==========================================
# 4. Advanced Sampling (SMOTE-NC)
# ==========================================

def apply_smote_nc(X_concat, y, cat_indices=None):
    """
    Applies SMOTE-NC (or SMOTE) to handle class imbalance.
    X_concat: numpy array of all features concatenated.
    y: numpy array of targets.
    """
    print("Applying SMOTE...")
    # Since we are using OHE, we strictly speaking have categorical features transformed.
    # Standard SMOTE treats them as continuous, which is a common approximation.
    # For true SMOTE-NC, we would need the raw data before OHE.
    # Here we use standard SMOTE on the OHE data.
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_concat, y)
    return X_res, y_res
