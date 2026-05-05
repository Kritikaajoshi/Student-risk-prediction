"""
03_preprocessing.py
-------------------
Builds a reusable preprocessing pipeline:
  - drops G1 and G2 (we want EARLY prediction, before mid-term grades)
  - drops G3 from features (it IS the target source)
  - one-hot encodes categoricals
  - scales numerics
  - returns train/test splits + the fitted pipeline

Saves:
  outputs/preprocessing_summary.txt
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student-mat.csv"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# Features kept for EARLY prediction. We deliberately exclude G1, G2:
# the goal is to flag risk before midterm grades are available.
EXCLUDE = {"G1", "G2", "G3", "at_risk"}

CATEGORICAL = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
NUMERIC = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def load_split(test_size=0.2, random_state=42):
    df = pd.read_csv(DATA, sep=";")
    df["at_risk"] = (df["G3"] < 10).astype(int)

    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feature_cols].copy()
    y = df["at_risk"].copy()

    # Stratify on target so train and test have the same risk rate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, df


def main():
    X_train, X_test, y_train, y_test, df = load_split()

    pre = build_preprocessor()
    pre.fit(X_train)

    # Save the preprocessor so it can be reused at scoring time
    joblib.dump(pre, MODELS / "preprocessor.joblib")

    summary = [
        f"Train rows: {len(X_train)}   At-risk: {y_train.sum()} ({y_train.mean():.1%})",
        f"Test rows:  {len(X_test)}    At-risk: {y_test.sum()} ({y_test.mean():.1%})",
        f"Numeric features ({len(NUMERIC)}): {NUMERIC}",
        f"Categorical features ({len(CATEGORICAL)}): {CATEGORICAL}",
        f"Total feature dims after one-hot: {pre.transform(X_train).shape[1]}",
        "",
        "Note: G1 and G2 (early grades) are deliberately EXCLUDED to ensure",
        "we are predicting risk from behavior + background, not from grades",
        "the student has already received in the same course.",
    ]
    text = "\n".join(summary)
    (OUT / "preprocessing_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
