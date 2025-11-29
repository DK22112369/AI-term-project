import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

CLEAN_PATH = Path("data/processed/us_accidents_clean.csv")
FEATURE_PATH = Path("data/processed/us_accidents_features.csv")

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """카테고리 컬럼들을 라벨/원핫 인코딩으로 변환."""
    print("Encoding categorical features...")
    # TODO: Cursor - implement categorical encoding (one-hot or target encoding)
    # Treat these columns as categorical: State, City, County, Weather_Condition, Street, Side.
    # Keep top 20, map rest to "Other".
    return df

def train_val_test_split_func(df: pd.DataFrame):
    """Stratified split으로 train/val/test 나누기."""
    print("Splitting data...")
    # TODO: Cursor - use train_test_split with stratify on Severity
    # Return df_train, df_val, df_test
    return df, df, df # Placeholder

def main():
    if not CLEAN_PATH.exists():
        print(f"Error: {CLEAN_PATH} not found. Run preprocess.py first.")
        return

    df = pd.read_csv(CLEAN_PATH)
    df = encode_categorical(df)
    
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving feature data to {FEATURE_PATH}...")
    df.to_csv(FEATURE_PATH, index=False)
    
    # Optional: Save splits if needed, or just demonstrate splitting
    train, val, test = train_val_test_split_func(df)
    print(f"Split sizes: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    print("Done.")

if __name__ == "__main__":
    main()
