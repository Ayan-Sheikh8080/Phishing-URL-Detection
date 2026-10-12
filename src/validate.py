import json
import sys
from pathlib import Path

import pandas as pd

from src.features import FEATURE_NAMES

DATA = Path("data/processed/features.csv")
REPORT = Path("reports/validation.json")
RATIO_COLS = ["digit_ratio", "letter_ratio"]


def validate(df: pd.DataFrame):
    errors = []
    expected = ["url"] + FEATURE_NAMES + ["label"]

    if df.empty:
        return ["Dataset is empty"], {}
    if list(df.columns) != expected:
        return [f"Schema mismatch. Expected {expected}, got {list(df.columns)}"], {}

    missing = int(df.isnull().sum().sum())
    dups = int(df.duplicated(subset="url").sum())
    bad_labels = int((~df["label"].isin([0, 1])).sum())
    negatives = int((df[FEATURE_NAMES] < 0).sum().sum())
    bad_ratios = int(((df[RATIO_COLS] < 0) | (df[RATIO_COLS] > 1)).sum().sum())
    phishing_share = float(df["label"].mean())

    if missing: errors.append(f"{missing} missing values")
    if dups: errors.append(f"{dups} duplicate URLs")
    if bad_labels: errors.append(f"{bad_labels} labels not in {{0,1}}")
    if negatives: errors.append(f"{negatives} negative feature values")
    if bad_ratios: errors.append(f"{bad_ratios} ratios outside [0,1]")
    if not 0.1 <= phishing_share <= 0.9:
        errors.append(f"Severe class imbalance: phishing share = {phishing_share:.2f}")

    stats = {
        "rows": len(df),
        "phishing_share": round(phishing_share, 4),
        "missing_values": missing,
        "duplicate_urls": dups,
        "invalid_labels": bad_labels,
    }
    return errors, stats


def main():
    errors, stats = validate(pd.read_csv(DATA))
    stats["passed"] = int(not errors)
    stats["num_errors"] = len(errors)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(stats, indent=2))

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("Validation passed:", stats)


if __name__ == "__main__":
    main()