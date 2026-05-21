"""Create a smaller working sample of the Kaggle fraud dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("creditcard.csv")
OUTPUT_PATH = Path("creditcard_small.csv")
LEGIT_SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(
            "creditcard.csv not found. Download the Kaggle dataset and place it in the project root."
        )

    df = pd.read_csv(INPUT_PATH)
    if "Class" not in df.columns:
        raise SystemExit("The dataset must contain a `Class` column.")

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    if fraud.empty:
        raise SystemExit("No fraud rows found. Check that you selected the correct dataset.")
    if legit.empty:
        raise SystemExit("No legitimate rows found. Check that you selected the correct dataset.")

    legit_sample_size = min(LEGIT_SAMPLE_SIZE, len(legit))
    legit_sample = legit.sample(n=legit_sample_size, random_state=RANDOM_STATE)

    small_df = (
        pd.concat([fraud, legit_sample], ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    small_df.to_csv(OUTPUT_PATH, index=False)

    print(
        f"Created {OUTPUT_PATH} with {len(small_df):,} rows "
        f"({len(fraud):,} fraud + {len(legit_sample):,} legit)."
    )


if __name__ == "__main__":
    main()
