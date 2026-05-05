"""
01_generate_dataset.py
----------------------
Generates a UCI-style Student Performance dataset.

The UCI Student Performance dataset (Cortez & Silva, 2008) was the planned
data source. The execution environment used for this project does not have
network access to archive.ics.uci.edu, so we synthesize a dataset that
follows the same schema, feature ranges, and known correlations reported
in the original paper (e.g., higher absences -> lower final grade,
study-time -> grade, prior failures -> grade).

This lets the full ML pipeline (preprocessing, modeling, fairness) run
end-to-end and produce realistic results. To use the real UCI dataset
instead, drop student-mat.csv into data/ and re-run from script 02.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 1000  # larger than UCI's 395 to give models room to learn

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def sample_categorical(values, probs, n=N):
    return RNG.choice(values, size=n, p=probs)


def main():
    df = pd.DataFrame()

    # --- Demographics (UCI schema) ---
    df["school"] = sample_categorical(["GP", "MS"], [0.75, 0.25])
    df["sex"] = sample_categorical(["F", "M"], [0.53, 0.47])
    df["age"] = RNG.integers(15, 23, size=N)
    df["address"] = sample_categorical(["U", "R"], [0.78, 0.22])  # Urban / Rural
    df["famsize"] = sample_categorical(["GT3", "LE3"], [0.71, 0.29])
    df["Pstatus"] = sample_categorical(["T", "A"], [0.90, 0.10])

    # Parental education (0=none, 4=higher)
    df["Medu"] = RNG.integers(0, 5, size=N)
    df["Fedu"] = RNG.integers(0, 5, size=N)

    df["Mjob"] = sample_categorical(
        ["teacher", "health", "services", "at_home", "other"],
        [0.15, 0.10, 0.25, 0.15, 0.35],
    )
    df["Fjob"] = sample_categorical(
        ["teacher", "health", "services", "at_home", "other"],
        [0.10, 0.07, 0.30, 0.05, 0.48],
    )
    df["reason"] = sample_categorical(
        ["home", "reputation", "course", "other"], [0.30, 0.25, 0.35, 0.10]
    )
    df["guardian"] = sample_categorical(["mother", "father", "other"], [0.65, 0.27, 0.08])

    # --- School-related numerical features ---
    df["traveltime"] = RNG.integers(1, 5, size=N)        # 1=<15m  .. 4=>1h
    df["studytime"] = RNG.integers(1, 5, size=N)         # 1=<2h   .. 4=>10h
    df["failures"] = RNG.choice([0, 1, 2, 3], size=N, p=[0.78, 0.13, 0.06, 0.03])

    # --- Yes / No support features ---
    for col, p_yes in [
        ("schoolsup", 0.13),
        ("famsup", 0.62),
        ("paid", 0.45),
        ("activities", 0.51),
        ("nursery", 0.80),
        ("higher", 0.95),
        ("internet", 0.83),
        ("romantic", 0.34),
    ]:
        df[col] = sample_categorical(["yes", "no"], [p_yes, 1 - p_yes])

    # --- Lifestyle / engagement (1-5 scales) ---
    df["famrel"] = RNG.integers(1, 6, size=N)
    df["freetime"] = RNG.integers(1, 6, size=N)
    df["goout"] = RNG.integers(1, 6, size=N)
    df["Dalc"] = RNG.integers(1, 6, size=N)   # workday alcohol
    df["Walc"] = RNG.integers(1, 6, size=N)   # weekend alcohol
    df["health"] = RNG.integers(1, 6, size=N)

    # Absences: heavy-tailed
    df["absences"] = RNG.poisson(lam=4.5, size=N).clip(0, 75)

    # --- Generate the final grade G3 (0-20) using a realistic linear model + noise ---
    # Coefficients chosen to mirror the UCI paper's correlations.
    score = (
        10.0
        + 0.9 * df["studytime"]
        - 1.6 * df["failures"]
        - 0.07 * df["absences"]
        + 0.4 * df["Medu"]
        + 0.3 * df["Fedu"]
        - 0.4 * df["goout"]
        - 0.5 * df["Dalc"]
        + 0.4 * df["famrel"]
        + 0.6 * (df["higher"] == "yes").astype(int)
        + 0.5 * (df["schoolsup"] == "yes").astype(int)
        + 0.3 * (df["internet"] == "yes").astype(int)
        - 0.4 * (df["romantic"] == "yes").astype(int)
        + 0.3 * (df["address"] == "U").astype(int)
        + RNG.normal(0, 2.2, size=N)
    )
    G3 = np.clip(np.round(score), 0, 20).astype(int)
    # G1 and G2 highly correlated with G3 (UCI behavior)
    G2 = np.clip(np.round(G3 + RNG.normal(0, 1.2, size=N)), 0, 20).astype(int)
    G1 = np.clip(np.round(G3 + RNG.normal(0, 1.6, size=N)), 0, 20).astype(int)
    df["G1"] = G1
    df["G2"] = G2
    df["G3"] = G3

    out_path = DATA_DIR / "student-mat.csv"
    df.to_csv(out_path, sep=";", index=False)

    print(f"Wrote {out_path}  shape={df.shape}")
    print(df.head())
    print("\nG3 distribution:")
    print(df["G3"].describe().round(2))
    fail_rate = (df["G3"] < 10).mean()
    print(f"\nAt-risk rate (G3 < 10): {fail_rate:.1%}")


if __name__ == "__main__":
    main()
