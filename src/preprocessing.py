from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Dataset path
DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "adult.data.txt"
)

CLEAN_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "preprocessed"
    / "adult_cleaned.csv"
)


# Dataset columns
columns = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income'
]


# Numerical attributes
numeric_cols = [
    'age',
    'fnlwgt',
    'education_num',
    'capital_gain',
    'capital_loss',
    'hours_per_week'
]


# Categorical attributes, including the target
categoric_cols = [
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native_country',
    'income'
]


def load_data(file_path=DATA_PATH):
    """Load the original Adult dataset."""

    df = pd.read_csv(
        file_path,
        header=None,
        names=columns,
    )

    return df


def clean_data(df):
    """Apply the data cleaning procedure from Part I."""

    clean = df.copy()

    # 1. Strip spaces
    for col in clean.columns:
        if not pd.api.types.is_numeric_dtype(clean[col]):
            clean[col] = clean[col].str.strip()

    # 2. Convert numeric columns
    for col in numeric_cols:
        clean[col] = pd.to_numeric(
            clean[col],
            errors='coerce'
        )

    # 3. Replace missing values with NaN
    clean = clean.replace(['?', ''], np.nan)
    clean = clean.replace('', np.nan)

    # 4. Standardize sex labels
    sex_fix = {
        'male': 'Male',
        'm': 'Male',
        'female': 'Female',
        'f': 'Female'
    }

    clean['sex'] = (
        clean['sex']
        .str.lower()
        .replace(sex_fix)
    )

    # 5. Drop rows containing missing values
    clean = clean.dropna().reset_index(drop=True)

    # 6. Remove duplicates after cleaning
    clean = clean.drop_duplicates().reset_index(drop=True)

    return clean


def split_data(clean):
    """Reproduce the stratified 85/15 split from Part I."""

    train, test = train_test_split(
        clean,
        test_size=0.15,
        random_state=42,
        stratify=clean['income']
    )

    return train, test


if __name__ == "__main__":
    data = load_data()
    clean = clean_data(data)

    CLEAN_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    clean.to_csv(
        CLEAN_DATA_PATH,
        index=False
    )

    print(f"Saved {clean.shape} to {CLEAN_DATA_PATH}")
