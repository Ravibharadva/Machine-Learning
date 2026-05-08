import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

def add_features(df):
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
    df['Title']      = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title']      = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr',
         'Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    df['Title']      = df['Title'].replace(
        {'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    return df

def get_preprocessor():
    numeric_features     = ['Age', 'Fare', 'FamilySize']
    categorical_features = ['Sex', 'Embarked', 'Title']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', numeric_transformer,     numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

def prepare_xy(df):
    df       = add_features(df)
    features = ['Age','Fare','FamilySize','IsAlone',
                'Pclass','Sex','Embarked','Title']
    X = df[features]
    y = df['Survived']
    return X, y