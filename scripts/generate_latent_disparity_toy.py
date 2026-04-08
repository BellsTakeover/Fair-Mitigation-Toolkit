from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_latent_disparity_toy(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Sensitive columns
    sex = rng.choice(["Female", "Male"], size=n, p=[0.5, 0.5])
    race = rng.choice(["A", "B", "C"], size=n, p=[0.45, 0.35, 0.20])
    ethnicity = rng.choice(["NH", "H"], size=n, p=[0.7, 0.3])

    # Hidden latent structural disadvantage
    # Strongly correlated with race/ethnicity, weakly with sex
    structural_risk = np.zeros(n)

    structural_risk += np.where(race == "A", -1.0, 0.0)
    structural_risk += np.where(race == "B", 0.25, 0.0)
    structural_risk += np.where(race == "C", 0.6, 0.0)

    structural_risk += np.where(ethnicity == "NH", -0.2, 0.0)
    structural_risk += np.where(ethnicity == "H", 0.8, 0.0)

    structural_risk += np.where(sex == "Male", 0.15, 0.0)
    structural_risk += rng.normal(0, 0.35, size=n)

    # Observable features that partly reflect the latent factor
    age = np.clip(48 + 5 * structural_risk + rng.normal(0, 8, size=n), 21, 85).round().astype(int)
    bmi = np.clip(26 + 1.8 * structural_risk + rng.normal(0, 2.0, size=n), 18, 45).round(2)
    waist = np.clip(90 + 5.5 * structural_risk + rng.normal(0, 5, size=n), 65, 150).round(1)
    hip = np.clip(101 + 1.2 * structural_risk + rng.normal(0, 4, size=n), 75, 160).round(1)

    # Proxy variables that hint at the latent variable
    # These are observed, but not the true latent itself
    access_score = np.clip(70 - 8 * structural_risk + rng.normal(0, 6, size=n), 0, 100).round(1)
    stress_score = np.clip(40 + 7 * structural_risk + rng.normal(0, 7, size=n), 0, 100).round(1)
    food_access_score = np.clip(75 - 10 * structural_risk + rng.normal(0, 7, size=n), 0, 100).round(1)

    # Outcome depends on both health features and the hidden latent factor
    logit = (
        -1.5
        + 0.035 * (age - 50)
        + 0.11 * (bmi - 27)
        + 0.035 * (waist - 92)
        - 0.010 * (hip - 102)
        - 0.018 * (access_score - 70)
        + 0.020 * (stress_score - 40)
        - 0.015 * (food_access_score - 75)
        + 0.4 * structural_risk
        + rng.normal(0, 0.6, size=n)
    )

    prob = sigmoid(logit)
    outcome = rng.binomial(1, prob, size=n)
    # after outcome is generated
    flip_mask = rng.random(n) < 0.1   # 10% noise
    outcome[flip_mask] = 1 - outcome[flip_mask]

    df = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "waist": waist,
            "hip": hip,
            "access_score": access_score,
            "stress_score": stress_score,
            "food_access_score": food_access_score,
            "sex": sex,
            "race": race,
            "ethnicity": ethnicity,
            "outcome": outcome,
        }
    )

    return df


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "toy_disparity_latent.csv"

    df = make_latent_disparity_toy(n=3000, seed=42)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Shape: {df.shape}")

    print("\nOutcome rate by sex:")
    print(df.groupby("sex")["outcome"].mean().round(3))

    print("\nOutcome rate by race:")
    print(df.groupby("race")["outcome"].mean().round(3))

    print("\nOutcome rate by ethnicity:")
    print(df.groupby("ethnicity")["outcome"].mean().round(3))

    print("\nClass balance:")
    print(df["outcome"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
