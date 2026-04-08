from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def save_dataset(df: pd.DataFrame, path: Path, name: str, notes: str) -> dict:
    df.to_csv(path, index=False)
    return {
        "name": name,
        "file": path.name,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "positive_rate": float(df["outcome"].mean()),
        "notes": notes,
    }


def make_sensitive_columns(rng: np.random.Generator, n: int):
    sex = rng.choice(["Female", "Male"], size=n, p=[0.5, 0.5])
    race = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.35, 0.25])
    ethnicity = rng.choice(["NH", "H"], size=n, p=[0.7, 0.3])
    return sex, race, ethnicity


def make_no_disparity(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sex, race, ethnicity = make_sensitive_columns(rng, n)

    h = rng.normal(0, 1, size=n)
    age = np.clip(50 + 8 * h + rng.normal(0, 2, size=n), 21, 85).round().astype(int)
    bmi = np.clip(27 + 2.0 * h + rng.normal(0, 0.6, size=n), 18, 45).round(2)
    waist = np.clip(92 + 6.0 * h + rng.normal(0, 1.5, size=n), 65, 150).round(1)
    hip = np.clip(102 + 1.5 * h + rng.normal(0, 1.5, size=n), 75, 160).round(1)

    score = (
        0.06 * (age - 50)
        + 0.23 * (bmi - 27)
        + 0.08 * (waist - 92)
        - 0.02 * (hip - 102)
        + rng.normal(0, 0.05, size=n)
    )
    outcome = (score > np.median(score)).astype(int)

    return pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "waist": waist,
        "hip": hip,
        "sex": sex,
        "race": race,
        "ethnicity": ethnicity,
        "outcome": outcome,
    })


def make_class_imbalance_smote(n: int = 3000, seed: int = 43) -> pd.DataFrame:
    """
    Goal:
    - strong global class imbalance
    - weak dependence on sensitive attributes
    - overlapping feature space
    - should give SMOTE a better chance
    """
    rng = np.random.default_rng(seed)
    sex, race, ethnicity = make_sensitive_columns(rng, n)

    z = rng.normal(0, 1, size=n)

    age = np.clip(49 + 5 * z + rng.normal(0, 4, size=n), 21, 85).round().astype(int)
    bmi = np.clip(27 + 1.8 * z + rng.normal(0, 1.3, size=n), 18, 45).round(2)
    waist = np.clip(91 + 5.0 * z + rng.normal(0, 2.8, size=n), 65, 150).round(1)
    hip = np.clip(102 + 0.8 * z + rng.normal(0, 2.6, size=n), 75, 160).round(1)

    # Tiny sensitive-attribute shifts only, so this is mostly a class imbalance problem
    small_group_shift = (
        np.where(race == "C", 0.04, 0.0)
        + np.where(ethnicity == "H", 0.03, 0.0)
        + np.where(sex == "Male", 0.01, 0.0)
    )

    logit = (
        -1.8
        + 0.030 * (age - 50)
        + 0.10 * (bmi - 27)
        + 0.025 * (waist - 92)
        - 0.008 * (hip - 102)
        + small_group_shift
        + rng.normal(0, 0.55, size=n)
    )
    prob = sigmoid(logit)
    outcome = rng.binomial(1, prob, size=n)

    return pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "waist": waist,
        "hip": hip,
        "sex": sex,
        "race": race,
        "ethnicity": ethnicity,
        "outcome": outcome,
    })


def make_label_noise_relabel(n: int = 3000, seed: int = 44) -> pd.DataFrame:
    """
    Goal:
    - classes roughly balanced
    - noise concentrated near decision boundary
    - little reason for SMOTE to dominate
    - should favor relabeling
    """
    rng = np.random.default_rng(seed)
    sex, race, ethnicity = make_sensitive_columns(rng, n)

    z = rng.normal(0, 1, size=n)

    age = np.clip(50 + 7 * z + rng.normal(0, 2, size=n), 21, 85).round().astype(int)
    bmi = np.clip(27 + 2.0 * z + rng.normal(0, 0.8, size=n), 18, 45).round(2)
    waist = np.clip(92 + 5.4 * z + rng.normal(0, 1.8, size=n), 65, 150).round(1)
    hip = np.clip(102 + 1.1 * z + rng.normal(0, 1.8, size=n), 75, 160).round(1)

    score = (
        0.055 * (age - 50)
        + 0.20 * (bmi - 27)
        + 0.07 * (waist - 92)
        - 0.015 * (hip - 102)
        + rng.normal(0, 0.10, size=n)
    )

    threshold = np.median(score)
    outcome = (score > threshold).astype(int)

    # Flip labels mainly around the boundary
    distance = np.abs(score - threshold)
    near_boundary = distance < np.quantile(distance, 0.22)
    flip_mask = near_boundary & (rng.random(n) < 0.45)
    outcome[flip_mask] = 1 - outcome[flip_mask]

    return pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "waist": waist,
        "hip": hip,
        "sex": sex,
        "race": race,
        "ethnicity": ethnicity,
        "outcome": outcome,
    })


def make_latent_disparity(n: int = 3000, seed: int = 45) -> pd.DataFrame:
    """
    Goal:
    - hidden latent factor exists
    - observable proxies reflect it
    - direct separation by sensitive attributes is moderate, not extreme
    - should give latent mitigation a better chance
    """
    rng = np.random.default_rng(seed)
    sex, race, ethnicity = make_sensitive_columns(rng, n)

    latent = np.zeros(n)
    latent += np.where(race == "A", -0.20, 0.0)
    latent += np.where(race == "B", 0.05, 0.0)
    latent += np.where(race == "C", 0.25, 0.0)
    latent += np.where(ethnicity == "H", 0.22, -0.04)
    latent += np.where(sex == "Male", 0.04, 0.0)
    latent += rng.normal(0, 0.40, size=n)

    age = np.clip(49 + 3.0 * latent + rng.normal(0, 8, size=n), 21, 85).round().astype(int)
    bmi = np.clip(26 + 1.3 * latent + rng.normal(0, 2.1, size=n), 18, 45).round(2)
    waist = np.clip(91 + 3.6 * latent + rng.normal(0, 4.0, size=n), 65, 150).round(1)
    hip = np.clip(101 + 0.8 * latent + rng.normal(0, 3.4, size=n), 75, 160).round(1)

    # Stronger proxies than before
    access_score = np.clip(70 - 11 * latent + rng.normal(0, 5.5, size=n), 0, 100).round(1)
    stress_score = np.clip(40 + 9 * latent + rng.normal(0, 5.5, size=n), 0, 100).round(1)
    food_access_score = np.clip(75 - 9 * latent + rng.normal(0, 5.5, size=n), 0, 100).round(1)

    logit = (
        -1.25
        + 0.028 * (age - 50)
        + 0.095 * (bmi - 27)
        + 0.022 * (waist - 92)
        - 0.009 * (hip - 102)
        - 0.018 * (access_score - 70)
        + 0.017 * (stress_score - 40)
        - 0.016 * (food_access_score - 75)
        + 0.14 * latent
        + rng.normal(0, 0.48, size=n)
    )
    prob = sigmoid(logit)
    outcome = rng.binomial(1, prob, size=n)

    return pd.DataFrame({
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
    })


def make_eo_targeted(n: int = 3000, seed: int = 46) -> pd.DataFrame:
    """
    Goal:
    - similar overall base rates across groups
    - but one group has noisier boundary / different error behavior
    - should help in-process equalized odds more than before
    """
    rng = np.random.default_rng(seed)
    sex, race, ethnicity = make_sensitive_columns(rng, n)

    z = rng.normal(0, 1, size=n)

    age = np.clip(50 + 6 * z + rng.normal(0, 3, size=n), 21, 85).round().astype(int)
    bmi = np.clip(27 + 1.8 * z + rng.normal(0, 1.1, size=n), 18, 45).round(2)
    waist = np.clip(92 + 5.0 * z + rng.normal(0, 2.1, size=n), 65, 150).round(1)
    hip = np.clip(102 + 0.9 * z + rng.normal(0, 2.1, size=n), 75, 160).round(1)

    base_logit = (
        -1.0
        + 0.040 * (age - 50)
        + 0.10 * (bmi - 27)
        + 0.028 * (waist - 92)
        - 0.010 * (hip - 102)
    )

    # Similar means, different noise by sex
    extra_noise = np.where(sex == "Male", rng.normal(0, 0.90, size=n), rng.normal(0, 0.35, size=n))

    logit = base_logit + extra_noise
    prob = sigmoid(logit)
    outcome = rng.binomial(1, prob, size=n)

    return pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "waist": waist,
        "hip": hip,
        "sex": sex,
        "race": race,
        "ethnicity": ethnicity,
        "outcome": outcome,
    })


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "validation_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = []

    datasets = [
        ("no_disparity.csv", make_no_disparity(), "Control dataset: should already be fair."),
        ("class_imbalance_smote.csv", make_class_imbalance_smote(), "Designed to validate SMOTE on mostly global class imbalance."),
        ("label_noise_relabel.csv", make_label_noise_relabel(), "Designed to validate relabeling on boundary label noise."),
        ("latent_disparity.csv", make_latent_disparity(), "Designed to validate latent mitigation with hidden structure plus proxies."),
        ("eo_targeted.csv", make_eo_targeted(), "Designed to validate in-process equalized odds on group-specific error behavior."),
    ]

    for filename, df, notes in datasets:
        meta = save_dataset(df, out_dir / filename, filename.replace(".csv", ""), notes)
        registry.append(meta)

    registry_df = pd.DataFrame(registry)
    registry_df.to_csv(out_dir / "dataset_registry.csv", index=False)

    print(f"Saved validation suite to: {out_dir}")
    print(registry_df.to_string(index=False))


if __name__ == "__main__":
    main()
