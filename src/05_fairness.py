"""
05_fairness.py
--------------
Fairness audit across demographic groups.

Metrics computed per group (sex, address, school):
  - selection_rate   : P(predicted at risk | group)
  - tpr              : recall within group  (equal opportunity)
  - fpr              : false alarm rate within group
  - precision        : within-group precision
  - n                : group size in test set

Disparity measures across the two groups for each attribute:
  - demographic_parity_diff = |sel_A - sel_B|
  - equal_opportunity_diff  = |tpr_A - tpr_B|
  - disparate_impact_ratio  = min(sel_A, sel_B) / max(sel_A, sel_B)
    (the "80% rule" — values >= 0.80 are typically considered acceptable)

Saves:
  outputs/fairness_per_group.csv
  outputs/fairness_disparities.csv
  figures/fairness_per_group.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"

PROTECTED = ["sex", "address", "school"]

sns.set_theme(style="whitegrid")


def per_group_metrics(df, model_col, group_col):
    rows = []
    for g, sub in df.groupby(group_col):
        y_true = sub["actual"].values
        y_pred = sub[model_col].values
        if len(np.unique(y_true)) < 2:
            tpr = float("nan")
            fpr = float("nan")
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        rows.append({
            "group_attr": group_col, "group": g, "n": len(sub),
            "selection_rate": y_pred.mean(),
            "tpr": tpr, "fpr": fpr,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "actual_rate": y_true.mean(),
        })
    return pd.DataFrame(rows)


def disparities(metrics_df):
    """Compute disparity measures per protected attribute, given two groups."""
    rows = []
    for attr, sub in metrics_df.groupby("group_attr"):
        if len(sub) != 2:
            continue
        a, b = sub.iloc[0], sub.iloc[1]
        sel_a, sel_b = a["selection_rate"], b["selection_rate"]
        tpr_a, tpr_b = a["tpr"], b["tpr"]
        fpr_a, fpr_b = a["fpr"], b["fpr"]
        di_ratio = (min(sel_a, sel_b) / max(sel_a, sel_b)) if max(sel_a, sel_b) > 0 else float("nan")
        rows.append({
            "attribute": attr,
            "group_A": a["group"], "group_B": b["group"],
            "selection_rate_A": sel_a, "selection_rate_B": sel_b,
            "demographic_parity_diff": abs(sel_a - sel_b),
            "tpr_A": tpr_a, "tpr_B": tpr_b,
            "equal_opportunity_diff": abs(tpr_a - tpr_b) if not np.isnan(tpr_a) and not np.isnan(tpr_b) else float("nan"),
            "fpr_A": fpr_a, "fpr_B": fpr_b,
            "fpr_diff": abs(fpr_a - fpr_b) if not np.isnan(fpr_a) and not np.isnan(fpr_b) else float("nan"),
            "disparate_impact_ratio": di_ratio,
            "passes_80pct_rule": di_ratio >= 0.80 if not np.isnan(di_ratio) else False,
        })
    return pd.DataFrame(rows)


def main():
    pred_df = pd.read_csv(OUT / "model_predictions.csv")

    # We focus the fairness audit on the model with best F1 + recall balance.
    # We use Logistic Regression as the "deployed" model because its recall
    # is highest (catches the most at-risk students, which matters for an
    # early-warning use case). Audit XGBoost too.
    target_models = {
        "Logistic Regression": "pred_Logistic Regression",
        "XGBoost": "pred_XGBoost",
    }

    all_per_group = []
    all_disparities = []

    for model_name, pred_col in target_models.items():
        for attr in PROTECTED:
            mg = per_group_metrics(pred_df, pred_col, attr)
            mg.insert(0, "model", model_name)
            all_per_group.append(mg)

            dis = disparities(mg)
            dis.insert(0, "model", model_name)
            all_disparities.append(dis)

    per_group_df = pd.concat(all_per_group, ignore_index=True)
    disp_df = pd.concat(all_disparities, ignore_index=True)

    per_group_df.to_csv(OUT / "fairness_per_group.csv", index=False)
    disp_df.to_csv(OUT / "fairness_disparities.csv", index=False)

    # --- Plot per-group selection rate and TPR for both models ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for r, model_name in enumerate(target_models.keys()):
        for c, attr in enumerate(PROTECTED):
            sub = per_group_df[(per_group_df.model == model_name) & (per_group_df.group_attr == attr)]
            x = np.arange(len(sub))
            ax = axes[r, c]
            w = 0.35
            ax.bar(x - w/2, sub["selection_rate"], width=w, label="P(flagged)", color="#3b82f6")
            ax.bar(x + w/2, sub["tpr"], width=w, label="TPR (recall)", color="#10b981")
            ax.set_xticks(x)
            ax.set_xticklabels(sub["group"].astype(str))
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{model_name} — by {attr}")
            for i, (s, t) in enumerate(zip(sub["selection_rate"], sub["tpr"])):
                ax.text(i - w/2, s + 0.02, f"{s:.0%}", ha="center", fontsize=8)
                ax.text(i + w/2, t + 0.02, f"{t:.0%}", ha="center", fontsize=8)
            if c == 0:
                ax.set_ylabel("Rate")
            if r == 0 and c == 2:
                ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fairness_per_group.png", dpi=140, bbox_inches="tight")
    plt.close()

    print("=== Per-group metrics ===")
    print(per_group_df.to_string(index=False))
    print("\n=== Disparity summary ===")
    print(disp_df.to_string(index=False))


if __name__ == "__main__":
    main()
