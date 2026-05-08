import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

SEED = 42
TEST_SIZE = 0.2

def preprocess(df, target):
    df = df.copy()
    #label-encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    #Median imputes any missing values
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    features_names = [c for c in df.columns if c != target]
    X = df[features_names].values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                random_state=SEED, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Features ({len(features_names)}): {features_names}")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")

    return X_train, X_test, y_train, y_test, features_names

