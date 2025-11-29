import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import os

# 1. Device Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 2. Data Loading
DATA_PATH = "data/raw/US_Accidents_March23.csv"
if not os.path.exists(DATA_PATH):
    # Try to find it in root if not in data/raw
    if os.path.exists("US_Accidents_March23.csv"):
        DATA_PATH = "US_Accidents_March23.csv"
    else:
        print(f"Error: {DATA_PATH} not found.")
        exit(1)

print(f"Loading dataset from {DATA_PATH}...")
# Use chunksize or just load it if memory allows. The notebook loaded it all.
# But 3GB might be large. However, the notebook sampled 10% immediately.
# Let's try loading it. If it fails, the user will know.
try:
    df = pd.read_csv(DATA_PATH, encoding='latin1')
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

print(f"Dataset loaded. Shape: {df.shape}")

# 3. Preprocessing
print("Preprocessing...")

# Filter for Severity 1 to 4
df = df[df['Severity'].isin([1, 2, 3, 4])].copy()

# Sample 10% stratified
if len(df) > 0 and len(df['Severity'].unique()) > 1:
    sampled_indices = df.groupby('Severity', group_keys=False).apply(
        lambda x: x.sample(frac=0.1, random_state=42), include_groups=False
    ).index
    df = df.loc[sampled_indices].copy()
    print(f"Sampled to {len(df)} rows.")

# Convert time columns
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
df.dropna(subset=['Start_Time', 'End_Time'], inplace=True)

# Feature Engineering
df['Start_Hour'] = df['Start_Time'].dt.hour
df['Start_DayOfWeek'] = df['Start_Time'].dt.dayofweek
df['Start_Month'] = df['Start_Time'].dt.month
df['Start_Year'] = df['Start_Time'].dt.year
df['Duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
df['Duration_min'] = df['Duration_min'].apply(lambda x: x if x > 0 else 0)

# Drop columns
columns_to_drop = [
    'ID', 'Source', 'Start_Time', 'End_Time', 'End_Lat', 'End_Lng',
    'Description', 'Street', 'City', 'County', 'Zipcode', 'Country',
    'Airport_Code', 'Weather_Timestamp', 'Start_Lat', 'Start_Lng', 'State',
    'Precipitation(in)', 'Wind_Chill(F)'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Impute missing values
numerical_cols = df.select_dtypes(include=np.number).columns
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("Preprocessing complete.")

# 4. Feature Grouping
TARGET = 'Severity'
y = df[TARGET].values - 1 # 0-indexed
df_features = df.drop(columns=[TARGET])

driver_features = [
    'Distance(mi)', 'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
    'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'
]

environment_features = [
    'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
    'Wind_Direction', 'Weather_Condition'
]

time_location_features = [
    'Timezone', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight',
    'Start_Hour', 'Start_DayOfWeek', 'Start_Month', 'Start_Year', 'Duration_min'
]

# Ensure features exist
driver_features = [f for f in driver_features if f in df_features.columns]
environment_features = [f for f in environment_features if f in df_features.columns]
time_location_features = [f for f in time_location_features if f in df_features.columns]

# Helper to get num/cat features
def get_num_cat(df, features):
    num = df[features].select_dtypes(include=np.number).columns.tolist()
    cat = df[features].select_dtypes(include=['object', 'bool', 'int32']).columns.tolist()
    # Note: int32 might be captured as num by default, but notebook treated some as cat.
    # The notebook logic:
    # time_location_categorical_new = df_new_features[time_location_features_new].select_dtypes(include=['object', 'bool', 'int32']).columns.tolist()
    # Let's stick to the notebook's logic if possible, or just rely on dtypes.
    # Start_Hour etc are int32/int64.
    return num, cat

driver_num, driver_cat = get_num_cat(df_features, driver_features)
env_num, env_cat = get_num_cat(df_features, environment_features)
# For time_loc, explicitly handle the int columns as categorical if needed, but let's trust the dtypes for now or force them.
# The notebook explicitly selected int32 for categorical in time_location.
# Let's check dtypes of time features. They are likely int64 or int32.
time_loc_num = df_features[time_location_features].select_dtypes(include=np.number).columns.tolist()
time_loc_cat = df_features[time_location_features].select_dtypes(include=['object', 'bool']).columns.tolist()
# Move Start_Hour, Start_DayOfWeek, Start_Month, Start_Year to categorical if they are in num
for col in ['Start_Hour', 'Start_DayOfWeek', 'Start_Month', 'Start_Year']:
    if col in time_loc_num:
        time_loc_num.remove(col)
        time_loc_cat.append(col)

# Transformers
driver_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), driver_num),
    ('cat', OneHotEncoder(handle_unknown='ignore'), driver_cat)
])

env_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), env_num),
    ('cat', OneHotEncoder(handle_unknown='ignore'), env_cat)
])

time_loc_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), time_loc_num),
    ('cat', OneHotEncoder(handle_unknown='ignore'), time_loc_cat)
])

print("Transforming features...")
X_driver = driver_preprocessor.fit_transform(df_features[driver_features])
X_env = env_preprocessor.fit_transform(df_features[environment_features])
X_time_loc = time_loc_preprocessor.fit_transform(df_features[time_location_features])

# Convert to dense
def to_dense(x):
    return x.toarray() if hasattr(x, 'toarray') else x

X_driver = to_dense(X_driver)
X_env = to_dense(X_env)
X_time_loc = to_dense(X_time_loc)

# Split
print("Splitting data...")
X_d_train, X_d_test, X_e_train, X_e_test, X_t_train, X_t_test, y_train, y_test = train_test_split(
    X_driver, X_env, X_time_loc, y, test_size=0.2, random_state=42, stratify=y
)

# Tensors
X_d_train_t = torch.tensor(X_d_train.astype(np.float32)).to(device)
X_e_train_t = torch.tensor(X_e_train.astype(np.float32)).to(device)
X_t_train_t = torch.tensor(X_t_train.astype(np.float32)).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

X_d_test_t = torch.tensor(X_d_test.astype(np.float32)).to(device)
X_e_test_t = torch.tensor(X_e_test.astype(np.float32)).to(device)
X_t_test_t = torch.tensor(X_t_test.astype(np.float32)).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

# 5. Model Definition
class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_size=64, dropout_rate=0.3):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return x

class CrashSeverityNet(nn.Module):
    def __init__(self, d_in, e_in, t_in, num_classes, block_hidden_size=64):
        super(CrashSeverityNet, self).__init__()
        self.d_block = MLPBlock(d_in, hidden_size=block_hidden_size)
        self.e_block = MLPBlock(e_in, hidden_size=block_hidden_size)
        self.t_block = MLPBlock(t_in, hidden_size=block_hidden_size)
        
        fusion_in = block_hidden_size * 3
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, xd, xe, xt):
        xd = self.d_block(xd)
        xe = self.e_block(xe)
        xt = self.t_block(xt)
        concat = torch.cat((xd, xe, xt), dim=1)
        return self.fusion_mlp(concat)

model = CrashSeverityNet(
    X_d_train.shape[1], X_e_train.shape[1], X_t_train.shape[1],
    len(np.unique(y))
).to(device)

print(model)

# 6. Training
train_dataset = TensorDataset(X_d_train_t, X_e_train_t, X_t_train_t, y_train_t)
test_dataset = TensorDataset(X_d_test_t, X_e_test_t, X_t_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10 # Reduced from 50 to 10 for "run at once" speed, or keep 50? 
# User said "run at once, I have to leave". 50 epochs might take too long if dataset is large.
# But 10% of 3GB is still ~300MB. 
# I'll stick to 10 epochs to be safe, or maybe 5. The notebook had 50.
# Let's do 20.
num_epochs = 20

print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for xd, xe, xt, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(xd, xe, xt)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# 7. Evaluation
print("Evaluating...")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for xd, xe, xt, labels in test_loader:
        outputs = model(xd, xe, xt)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds))

# Save model
os.makedirs("models", exist_ok=True)
save_path = "models/crash_severity_net_new.pt"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
