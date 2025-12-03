import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, fbeta_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import joblib

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_size=64, dropout_rate=0.3):
        super(MLPBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)

class CrashSeverityNet(nn.Module):
    def __init__(self, input_dims, num_classes, block_hidden_size=64):
        super(CrashSeverityNet, self).__init__()
        self.encoders = nn.ModuleDict()
        for group_name, dim in input_dims.items():
            self.encoders[group_name] = MLPBlock(dim, hidden_size=block_hidden_size)
        fusion_in = block_hidden_size * len(input_dims)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, inputs):
        encoded_features = []
        for group_name in sorted(self.encoders.keys()):
            if group_name in inputs:
                x = inputs[group_name]
                out = self.encoders[group_name](x)
                encoded_features.append(out)
        concat = torch.cat(encoded_features, dim=1)
        return self.fusion_mlp(concat)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def load_full_dataset(path):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding_errors='replace')
    return df

def clean_and_engineer_features(df):
    df = df.copy()
    df = df.dropna(subset=['Severity'])
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    df = df.dropna(subset=['Start_Time'])
    df['Start_Hour'] = df['Start_Time'].dt.hour
    df['Start_DayOfWeek'] = df['Start_Time'].dt.dayofweek
    df['Start_Month'] = df['Start_Time'].dt.month
    df['Duration_minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60.0
    df['Duration_minutes'] = df['Duration_minutes'].fillna(0)
    road_features = ['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
    for col in road_features:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df

def fit_feature_transformers(df_train):
    feature_groups = {
        'temporal': ['Start_Hour', 'Start_DayOfWeek', 'Start_Month', 'Duration_minutes'],
        'weather': ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition'],
        'road': ['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'],
        'spatial': ['Start_Lat', 'Start_Lng', 'Distance(mi)']
    }
    transformers = {}
    for group_name, features in feature_groups.items():
        valid_features = [f for f in features if f in df_train.columns]
        num_cols = [f for f in valid_features if pd.api.types.is_numeric_dtype(df_train[f])]
        cat_cols = [f for f in valid_features if not pd.api.types.is_numeric_dtype(df_train[f])]
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

def transform_with_preprocessors(df, bundle, device="cpu"):
    outputs = {}
    for group_name in ['temporal', 'weather', 'road', 'spatial']:
        preprocessor = bundle.get(f"{group_name}_preprocessor")
        if preprocessor:
            X = preprocessor.transform(df)
            if hasattr(X, 'toarray'):
                X = X.toarray()
            outputs[group_name] = torch.tensor(X.astype(np.float32)).to(device)
        else:
            outputs[group_name] = torch.tensor([]).to(device)
    y = df["Severity"].values - 1
    y_t = torch.tensor(y).long().to(device)
    return outputs['temporal'], outputs['weather'], outputs['road'], outputs['spatial'], y_t

def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_f2 = fbeta_score(y_true, y_pred, beta=2, average='macro')
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_f2": macro_f2,
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/raw/US_Accidents_March23.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but not available. Please enable GPU support.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    
    if not os.path.exists(args.data_path):
        print(f"Dataset not found at {args.data_path}")
        return

    df = load_full_dataset(args.data_path)
    if args.sample_frac < 1.0:
        df = df.groupby('Severity', group_keys=False).apply(lambda x: x.sample(frac=args.sample_frac, random_state=args.seed))
    
    df = clean_and_engineer_features(df)
    
    df = df.sort_values('Start_Time')
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    bundle = fit_feature_transformers(df_train)
    
    X_t_train, X_w_train, X_r_train, X_s_train, y_train = transform_with_preprocessors(df_train, bundle, args.device)
    X_t_val, X_w_val, X_r_val, X_s_val, y_val = transform_with_preprocessors(df_val, bundle, args.device)
    X_t_test, X_w_test, X_r_test, X_s_test, y_test = transform_with_preprocessors(df_test, bundle, args.device)

    train_dataset = TensorDataset(X_t_train, X_w_train, X_r_train, X_s_train, y_train)
    val_dataset = TensorDataset(X_t_val, X_w_val, X_r_val, X_s_val, y_val)
    test_dataset = TensorDataset(X_t_test, X_w_test, X_r_test, X_s_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    input_dims = {
        'temporal': X_t_train.shape[1],
        'weather': X_w_train.shape[1],
        'road': X_r_train.shape[1],
        'spatial': X_s_train.shape[1]
    }

    model = CrashSeverityNet(input_dims, 4).to(args.device)
    
    classes = np.unique(y_train.cpu().numpy())
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train.cpu().numpy())
    class_weights = torch.FloatTensor(weights).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for xt, xw, xr, xs, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            xt, xw, xr, xs, y = xt.to(args.device), xw.to(args.device), xr.to(args.device), xs.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
            out = model(inputs)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xt, xw, xr, xs, y in val_loader:
                xt, xw, xr, xs, y = xt.to(args.device), xw.to(args.device), xr.to(args.device), xs.to(args.device), y.to(args.device)
                inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
                out = model(inputs)
                loss = criterion(out, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")

    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for xt, xw, xr, xs, y in test_loader:
            xt, xw, xr, xs, y = xt.to(args.device), xw.to(args.device), xr.to(args.device), xs.to(args.device), y.to(args.device)
            inputs = {'temporal': xt, 'weather': xw, 'road': xr, 'spatial': xs}
            out = model(inputs)
            preds = torch.argmax(out, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(y.cpu().numpy())

    metrics = evaluate_metrics(test_targets, test_preds)
    print("\nTest Results:")
    print(json.dumps(metrics['classification_report'], indent=4))
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

if __name__ == "__main__":
    main()
