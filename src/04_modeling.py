"""
04_modeling.py
--------------
Trains and evaluates 5 models for at-risk prediction.

Models:
  1. Logistic Regression (baseline, interpretable)
  2. K-Nearest Neighbors  (baseline)
  3. Random Forest        (ensemble)
  4. Gradient Boosting    (ensemble)
  5. XGBoost              (gradient boosting)

For each model:
  - tunes hyperparameters with 5-fold stratified CV on the TRAIN set,
    optimizing F1 (good for imbalanced classes)
  - evaluates the best estimator on the held-out TEST set
  - records: Accuracy, Precision, Recall, F1, ROC-AUC, MCC, confusion matrix

Saves:
  outputs/model_comparison.csv
  outputs/model_predictions.csv  (test-set probs for all models, used downstream)
  models/best_<name>.joblib
  figures/model_roc_curves.png
  figures/model_confusion_matrices.png
  figures/model_metric_bars.png
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import sys
sys.path.append(str(Path(__file__).parent))
from importlib import import_module
prep_module = import_module("03_preprocessing")
load_split = prep_module.load_split
build_preprocessor = prep_module.build_preprocessor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
MODELS = ROOT / "models"
for d in (OUT, FIG, MODELS):
    d.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def make_search(name, estimator, param_grid):
    """Wrap estimator in a Pipeline with the preprocessor and run grid search."""
    pipe = Pipeline([("pre", build_preprocessor()), ("clf", estimator)])
    grid = {f"clf__{k}": v for k, v in param_grid.items()}
    return GridSearchCV(pipe, grid, cv=CV, scoring="f1", n_jobs=-1, refit=True)


def get_search_specs():
    """Return dict of name -> (estimator, param_grid)."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            {"C": [0.1, 1.0, 10.0], "penalty": ["l2"]},
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {"n_neighbors": [5, 11, 21, 31], "weights": ["uniform", "distance"]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 3]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [150, 300], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss", random_state=42, n_jobs=-1,
                # scale_pos_weight handles imbalance; calculated per fold in fit
                scale_pos_weight=(1 - 0.171) / 0.171,
            ),
            {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        ),
    }


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1],
    }, y_pred, y_prob


def plot_roc_curves(rocs, y_test, save):
    plt.figure(figsize=(7, 6))
    for name, y_prob in rocs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — At-Risk Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save, dpi=140, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(preds, y_test, save):
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6))
    for ax, (name, y_pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Not", "At risk"], yticklabels=["Not", "At risk"])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save, dpi=140, bbox_inches="tight")
    plt.close()


def plot_metric_bars(df, save):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
    plot_df = df.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")
    plt.figure(figsize=(11, 5))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model")
    plt.ylim(0, 1.05)
    plt.title("Model performance on held-out test set")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(save, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    X_train, X_test, y_train, y_test, df = load_split()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    rows, preds_dict, probs_dict = [], {}, {}
    for name, (est, grid) in get_search_specs().items():
        print(f"\n>>> Tuning {name} ...")
        search = make_search(name, est, grid)
        search.fit(X_train, y_train)
        print(f"    best CV F1: {search.best_score_:.3f}")
        print(f"    best params: {search.best_params_}")

        metrics, y_pred, y_prob = evaluate(name, search.best_estimator_, X_test, y_test)
        rows.append(metrics)
        preds_dict[name] = y_pred
        probs_dict[name] = y_prob

        joblib.dump(search.best_estimator_, MODELS / f"best_{name.replace(' ', '_').lower()}.joblib")

    results = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    results.to_csv(OUT / "model_comparison.csv", index=False)

    # Save test-set indices + predictions for downstream fairness analysis
    pred_df = X_test.copy()
    pred_df["actual"] = y_test.values
    for name, probs in probs_dict.items():
        pred_df[f"prob_{name}"] = probs
        pred_df[f"pred_{name}"] = preds_dict[name]
    pred_df.to_csv(OUT / "model_predictions.csv", index=False)

    plot_roc_curves(probs_dict, y_test, FIG / "model_roc_curves.png")
    plot_confusion_matrices(preds_dict, y_test, FIG / "model_confusion_matrices.png")
    plot_metric_bars(results, FIG / "model_metric_bars.png")

    print("\n=== Final results (sorted by F1) ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
