from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def make_no_disparity_toy(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Builds a synthetic dataset where:
    - sensitive groups are balanced
    - the label is driven by health features, not by sex/race/ethnicity
    - baseline fairness should stay relatively low compared with a disparity dataset
    """

    rng = np.random.default_rng(seed)

    # Balanced sensitive attributes
    sex = np.array(["Female", "Male"] * (n // 2) + ["Female"] * (n % 2), dtype=object)
    rng.shuffle(sex)

    race = np.tile(np.array(["A", "B", "C"], dtype=object), int(np.ceil(n / 3)))[:n]
    rng.shuffle(race)

    ethnicity = np.array(["H", "NH"] * (n // 2) + ["H"] * (n % 2), dtype=object)
    rng.shuffle(ethnicity)

    # Shared latent health score that is NOT based on sensitive columns
    h = rng.normal(0, 1, size=n)

    age = np.clip(50 + 10 * h + rng.normal(0, 1.5, size=n), 21, 80).round().astype(int)
    bmi = np.clip(27 + 2.2 * h + rng.normal(0, 0.4, size=n), 18, 40).round(2)
    waist = np.clip(92 + 6.5 * h + rng.normal(0, 1.2, size=n), 65, 140).round(1)
    hip = np.clip(102 + 1.5 * h + rng.normal(0, 1.2, size=n), 75, 150).round(1)

    # Outcome depends on health features only
    score = (
        0.06 * (age - 50)
        + 0.25 * (bmi - 27)
        + 0.09 * (waist - 92)
        - 0.02 * (hip - 102)
        + rng.normal(0, 0.18, size=n)
    )

    # Use the median to keep the dataset roughly balanced
    threshold = float(np.median(score))
    outcome = (score > threshold).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "waist": waist,
            "hip": hip,
            "sex": sex,
            "race": race,
            "ethnicity": ethnicity,
            "outcome": outcome,
        }
    )

    return df


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "toy_no_disparity.csv"

    df = make_no_disparity_toy(n=3000, seed=42)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Shape: {df.shape}")
    print("\nOutcome rate by sex:")
    print(df.groupby("sex")["outcome"].mean())
    print("\nOutcome rate by race:")
    print(df.groupby("race")["outcome"].mean())
    print("\nOutcome rate by ethnicity:")
    print(df.groupby("ethnicity")["outcome"].mean())


if __name__ == "__main__":
    main()
