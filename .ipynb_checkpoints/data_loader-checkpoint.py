import numpy as np
import pandas as pd

SEED = 42

def load_heart_disease(path="heart.xls"):
    df = pd.read_csv(path)
    print(f"[heart] Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target distribution: \n{df['HeartDisease'].value_counts().to_string()}\n")
    return df, "HeartDisease"
    

