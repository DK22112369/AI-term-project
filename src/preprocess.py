import pandas as pd
from pathlib import Path
import numpy as np

# Define paths relative to the project root (assuming script is run from root or src)
# Adjust if necessary based on execution context
RAW_PATH = Path("data/raw/US_Accidents_March23.csv")
CLEAN_PATH = Path("data/processed/us_accidents_clean.csv")

USE_COLUMNS = [
    "Severity",
    "Start_Time",
    "Start_Lat", "Start_Lng",
    "City", "County", "State",
    "Distance(mi)",
    "Temperature(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
    "Weather_Condition",
    "Street", "Side",
    "Amenity", "Bump", "Crossing", "Junction",
    "Railway", "Stop", "Traffic_Signal", "Roundabout",
]

def load_raw():
    print(f"Loading data from {RAW_PATH}...")
    return pd.read_csv(RAW_PATH)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """US Accidents 전처리의 1단계: 컬럼 선택, Severity 필터링, 기본 결측 처리.
    TODO: Cursor에게 내부 구현 요구
    """
    print("Applying basic cleaning...")
    # TODO: implement filtering Severity in [1,2,3] and selecting USE_COLUMNS
    # TODO: convert Start_Time to datetime, handle obvious missing values
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Start_Time에서 연/월/요일/시간/주말/야간 파생변수 생성."""
    print("Adding time features...")
    # TODO: implement year, month, dayofweek, hour, is_weekend, is_night
    return df

def handle_outliers_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 이상치 클리핑 및 결측값 처리."""
    print("Handling outliers and missing values...")
    # TODO: implement numeric clipping and imputation
    return df

def main():
    # Ensure we are in the right directory or paths are correct
    if not RAW_PATH.exists():
        print(f"Error: {RAW_PATH} not found. Please run from project root.")
        return

    df = load_raw()
    df = basic_clean(df)
    df = add_time_features(df)
    df = handle_outliers_and_missing(df)
    
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving cleaned data to {CLEAN_PATH}...")
    df.to_csv(CLEAN_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
