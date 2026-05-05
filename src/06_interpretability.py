"""
06_interpretability.py
----------------------
Generates feature-importance and SHAP visualizations for the best models.

Saves:
  figures/feature_importance_xgb.png
  figures/feature_importance_logreg.png
  figures/shap_summary_xgb.png
  outputs/feature_importance.csv
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

import sys
sys.path.append(str(Path(__file__).parent))
from importlib import import_module
prep = import_module("03_preprocessing")

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
MODELS = ROOT / "models"


def get_feature_names(pipeline):
    """Get feature names after one-hot encoding."""
    pre = pipeline.named_steps["pre"]
    return pre.get_feature_names_out()


def main():
    X_train, X_test, y_train, y_test, _ = prep.load_split()

    # --- XGBoost feature importance + SHAP ---
    xgb_pipe = joblib.load(MODELS / "best_xgboost.joblib")
    feat_names = [n.split("__", 1)[1] for n in get_feature_names(xgb_pipe)]

    xgb_clf = xgb_pipe.named_steps["clf"]
    importances = xgb_clf.feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance_xgb": importances})
    imp_df = imp_df.sort_values("importance_xgb", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(imp_df["feature"][::-1], imp_df["importance_xgb"][::-1], color="#7c3aed")
    ax.set_title("XGBoost feature importance (top 20)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIG / "feature_importance_xgb.png", dpi=140, bbox_inches="tight")
    plt.close()

    # --- Logistic Regression coefficients ---
    lr_pipe = joblib.load(MODELS / "best_logistic_regression.joblib")
    lr_feat_names = [n.split("__", 1)[1] for n in get_feature_names(lr_pipe)]
    coefs = lr_pipe.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({"feature": lr_feat_names, "coef_logreg": coefs})
    # Sort by absolute value, take top 20
    coef_df["abs"] = coef_df["coef_logreg"].abs()
    coef_top = coef_df.sort_values("abs", ascending=False).head(20).drop(columns="abs")

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#ef4444" if c > 0 else "#10b981" for c in coef_top["coef_logreg"][::-1]]
    ax.barh(coef_top["feature"][::-1], coef_top["coef_logreg"][::-1], color=colors)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title("Logistic Regression coefficients (top 20 by |coef|)\n"
                 "Red = increases at-risk probability,  Green = decreases it")
    ax.set_xlabel("Standardized coefficient")
    plt.tight_layout()
    plt.savefig(FIG / "feature_importance_logreg.png", dpi=140, bbox_inches="tight")
    plt.close()

    # --- SHAP summary plot for XGBoost ---
    X_test_processed = xgb_pipe.named_steps["pre"].transform(X_test)
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_test_processed)

    plt.figure()
    shap.summary_plot(shap_values, X_test_processed, feature_names=feat_names,
                      max_display=15, show=False, plot_size=(9, 6))
    plt.tight_layout()
    plt.savefig(FIG / "shap_summary_xgb.png", dpi=140, bbox_inches="tight")
    plt.close()

    # --- Save combined importance table ---
    merged = imp_df.merge(coef_df.drop(columns="abs"), on="feature", how="outer")
    merged.to_csv(OUT / "feature_importance.csv", index=False)

    print("Top XGBoost features:")
    print(imp_df.head(10).to_string(index=False))
    print("\nTop Logistic Regression features by |coef|:")
    print(coef_top.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
