import os
import sys

# Import necessary libraries at the top
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from joblib import dump, load

# CONFIG ?뱀뀡: CSV ?뚯씪 寃쎈줈 ?ㅼ젙怨?湲고? 湲곕낯 ?ㅼ젙
DATA_PATH = "US_Accidents_March23.csv"
MODEL_PATH = "models/severity_rf.joblib"
TARGET_COL = "Severity"

# Feature definitions
numeric_features = [
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)"
]

categorical_features = [
    "Weather_Condition",
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight"
]

# ?붾쾭洹?異쒕젰: ?꾩옱 ?묒뾽 ?붾젆?곕━? ?ㅼ젣 李얜뒗 寃쎈줈
print(f"[DEBUG] Current working directory: {os.getcwd()}")
print(f"[DEBUG] Full path to DATA_PATH: {os.path.abspath(DATA_PATH)}")

# ?뚯씪 議댁옱 ?щ? 泥댄겕 諛??먮윭 泥섎━
if not os.path.exists(DATA_PATH):
    print("[ERROR] CSV file not found at the specified path.")
    print("Check the following information:")
    print(f"  Current working directory: {os.getcwd()}")
    print(f"  DATA_PATH value: {DATA_PATH}")
    print(f"  Full path to DATA_PATH: {os.path.abspath(DATA_PATH)}")
    sys.exit(1)

# ?곗씠??濡쒕뱶 諛?異뺤냼
def load_and_reduce_data(file_path: str, keep_ratio: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    print(f"[INFO] Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} column not found.")
    df = df[df[TARGET_COL].notna()]
    # keep_ratio 留뚰겮留??좎? (stratify濡?Severity 鍮꾩쑉 ?좎?)
    df_small, _ = train_test_split(
        df,
        test_size=1 - keep_ratio,
        stratify=df[TARGET_COL],
        random_state=random_state
    )
    print(f"[INFO] Reduced shape: {df_small.shape}")
    return df_small

# ?꾩쿂由?+ 紐⑤뜽 ?뚯씠?꾨씪??
def build_pipeline(random_state: int = 42) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1
    )
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", rf)
    ])
    return pipe

# ?숈뒿 + ?됯? + ???
def train_evaluate_and_save(df: pd.DataFrame, model_path: str = MODEL_PATH, test_size: float = 0.2, random_state: int = 42) -> Pipeline:
    X = df[numeric_features + categorical_features]
    y = df[TARGET_COL]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"[INFO] Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    pipe = build_pipeline(random_state=random_state)
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5]
    }
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )
    print("[INFO] Start training with GridSearchCV...")
    grid.fit(X_train, y_train)
    best_model: Pipeline = grid.best_estimator_
    print(f"[INFO] Best params: {grid.best_params_}")
    # ?됯?
    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"[RESULT] Validation Accuracy: {acc:.4f}")
    print("[RESULT] Classification Report:")
    print(classification_report(y_val, y_pred))
    # ROC-AUC (?댁쭊 遺꾨쪟???뚮쭔)
    if hasattr(best_model, "predict_proba") and len(best_model.classes_) == 2:
        y_proba = best_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        print(f"[RESULT] ROC-AUC Score: {auc:.4f}")
    else:
        print("[INFO] ROC-AUC is this binary classification only. (Currently multi-class classification)")
    # ???
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(best_model, model_path)
    print(f"[INFO] Model saved to: {model_path}")
    return best_model

# 紐⑤뜽 濡쒕뱶 / ?명띁?곗뒪
def load_model(model_path: str) -> Pipeline:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"[INFO] Loading model from: {model_path}")
    return load(model_path)

def predict_severity(model: Pipeline, input_df: pd.DataFrame):
    X_input = input_df[numeric_features + categorical_features]
    preds = model.predict(X_input)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)
    return preds, probs

# ?덉떆 ?ㅽ뻾 (吏곸젒 ?뚮젮蹂닿퀬 ?띠쓣 ??
if __name__ == "__main__":
    # ?곗씠??寃쎈줈 ?ㅼ젙 ?뺣━
    DATA_PATH = "US_Accidents_March23.csv"
