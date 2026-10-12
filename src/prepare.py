from pathlib import Path

import pandas as pd

from src.features import extract_features

RAW = Path("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
OUT = Path("data/processed/features.csv")


def main():
    df = pd.read_csv(RAW, usecols=["URL", "label"])
    n_raw = len(df)

    df = df.dropna(subset=["URL", "label"])
    df = df.drop_duplicates(subset="URL").reset_index(drop=True)
    df["label"] = 1 - df["label"].astype(int)  # raw: 1 = legit, 0 = phishing

    feats = pd.DataFrame([extract_features(u) for u in df["URL"]])
    out = pd.concat(
        [df[["URL"]].rename(columns={"URL": "url"}), feats, df[["label"]]], axis=1
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Raw rows: {n_raw} | kept: {len(out)} | dropped: {n_raw - len(out)}")
    print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()