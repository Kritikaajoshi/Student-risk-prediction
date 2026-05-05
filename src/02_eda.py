"""
02_eda.py
---------
Exploratory Data Analysis for the student dataset.

Produces:
  figures/eda_target_distribution.png
  figures/eda_correlation_heatmap.png
  figures/eda_feature_target.png
  figures/eda_demographic_risk.png
  outputs/eda_summary.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student-mat.csv"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def main():
    df = pd.read_csv(DATA, sep=";")
    df["at_risk"] = (df["G3"] < 10).astype(int)

    summary = []
    summary.append(f"Rows: {len(df)}")
    summary.append(f"Columns: {df.shape[1]}")
    summary.append(f"At-risk students: {df['at_risk'].sum()} ({df['at_risk'].mean():.1%})")
    summary.append(f"Missing values: {df.isna().sum().sum()}")

    # --- Target distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["G3"], bins=20, ax=axes[0], color="#3b82f6")
    axes[0].axvline(10, color="red", ls="--", label="At-risk cutoff (G3 < 10)")
    axes[0].set_title("Final Grade (G3) Distribution")
    axes[0].set_xlabel("G3 (0-20)")
    axes[0].legend()

    counts = df["at_risk"].value_counts().rename({0: "Not at risk", 1: "At risk"})
    counts.plot(kind="bar", ax=axes[1], color=["#10b981", "#ef4444"])
    axes[1].set_title("Class Balance (binary target)")
    axes[1].set_ylabel("Students")
    axes[1].set_xticklabels(counts.index, rotation=0)
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 5, f"{v}\n({v/len(df):.1%})", ha="center")
    plt.tight_layout()
    plt.savefig(FIG / "eda_target_distribution.png", dpi=140, bbox_inches="tight")
    plt.close()

    # --- Correlation heatmap (numeric features only, drop G1/G2 to avoid leakage view) ---
    numeric = df.select_dtypes(include=np.number).drop(columns=["G1", "G2"])
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Correlation Matrix (numeric features, no G1/G2)")
    plt.tight_layout()
    plt.savefig(FIG / "eda_correlation_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close()

    # --- Top correlated features with G3 ---
    g3_corr = corr["G3"].drop(["G3", "at_risk"]).sort_values(key=abs, ascending=False)
    summary.append("\nTop correlations with G3 (no G1/G2):")
    summary.append(g3_corr.head(10).to_string())

    # --- Engagement features vs at-risk rate ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, ["studytime", "failures", "absences", "goout"]):
        if col == "absences":
            df["abs_bin"] = pd.cut(df["absences"], bins=[-1, 2, 5, 10, 20, 100],
                                   labels=["0-2", "3-5", "6-10", "11-20", "20+"])
            grouped = df.groupby("abs_bin", observed=True)["at_risk"].mean()
        else:
            grouped = df.groupby(col)["at_risk"].mean()
        grouped.plot(kind="bar", ax=ax, color="#6366f1")
        ax.set_title(f"At-risk rate vs {col}")
        ax.set_ylabel("P(at risk)")
        ax.set_ylim(0, max(0.8, grouped.max() * 1.15))
        for i, v in enumerate(grouped.values):
            ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "eda_feature_target.png", dpi=140, bbox_inches="tight")
    plt.close()

    # --- Demographic breakdown for fairness preview ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ["sex", "address", "school"]):
        rates = df.groupby(col)["at_risk"].mean()
        rates.plot(kind="bar", ax=ax, color="#0ea5e9")
        ax.set_title(f"At-risk rate by {col}")
        ax.set_ylabel("P(at risk)")
        ax.set_xticklabels(rates.index, rotation=0)
        for i, v in enumerate(rates.values):
            ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "eda_demographic_risk.png", dpi=140, bbox_inches="tight")
    plt.close()

    summary.append("\nDemographic at-risk rates:")
    for col in ["sex", "address", "school"]:
        summary.append(f"  {col}:")
        for k, v in df.groupby(col)["at_risk"].mean().items():
            summary.append(f"    {k}: {v:.1%}")

    text = "\n".join(summary)
    (OUT / "eda_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
