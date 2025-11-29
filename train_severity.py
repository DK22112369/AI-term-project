import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# 데이터 / 모델 경로 설정
DATA_PATH = "US_Accidents_March23.csv"
MODEL_PATH = "models/severity_rf.joblib"
TARGET_COL = "Severity"


def load_and_reduce_data(file_path, keep_ratio=0.2, random_state=42):
    """
    CSV 파일을 로드하고, Severity 기준으로 일부만 샘플링해서
    데이터 크기를 줄입니다.
    """
    print(f"[INFO] Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} 컬럼을 찾을 수 없습니다.")

    # 여기서는 예시로 Severity 1,2,3만 사용 + 전체의 keep_ratio만 샘플링
    # 필요하면 isin 부분은 제거하고 전체에서 샘플링해도 됩니다.
    df = df[df[TARGET_COL].isin([1, 2, 3])]
    df_small = df.sample(frac=keep_ratio, random_state=random_state)

    print(f"[INFO] Original shape: {df.shape}, Reduced shape: {df_small.shape}")
    return df_small


def train_evaluate_and_save(
    df,
    model_path=MODEL_PATH,
    test_size=0.2,
    random_state=42
):
    """
    - df를 train/test로 나눠서 RandomForest를 학습
    - Accuracy, Classification Report, ROC-AUC 출력
    - 모델을 model_path에 저장
    """
    print("[INFO] Start train_evaluate_and_save")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # RandomForest 모델 정의 및 학습
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )
    print("[INFO] Training RandomForest...")
    model.fit(X_train, y_train)

    # 평가
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Accuracy: {acc:.4f}")

    print("[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred))

    # ROC-AUC (이진 분류일 때만 의미 있음, 여기서는 일단 multi-class로 시도)
    try:
        if len(model.classes_) == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
            print(f"[RESULT] ROC-AUC: {roc:.4f}")
        else:
            print("[INFO] ROC-AUC는 이진 분류에서만 정확하게 해석 가능 (현재 다중 클래스).")
    except Exception as e:
        print(f"[WARN] ROC-AUC 계산 중 오류 발생: {e}")

    # 모델 저장
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[INFO] Model saved to: {model_path}")

        
def main():
    # 1) 데이터 로드 + 축소
    df_small = load_and_reduce_data(DATA_PATH, keep_ratio=0.2)

    # 2) 학습 + 평가 + 모델 저장
    train_evaluate_and_save(df_small, MODEL_PATH)

    print("[DONE] Training finished and model saved.")


if __name__ == "__main__":
    main()
