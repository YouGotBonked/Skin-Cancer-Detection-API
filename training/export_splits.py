"""Export the train/val/test splits from the database to a CSV.

The CSV is all the training script needs, so you can train anywhere (e.g. Colab)
without putting your database password there.

Usage:
    python export_splits.py                 # writes data/splits.csv
    python export_splits.py --out my.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from db import SessionLocal
from models import ImageRecord, Lesion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("../data/splits.csv"))
    args = ap.parse_args()

    with SessionLocal() as db:
        rows = db.execute(
            select(
                ImageRecord.external_image_id,
                ImageRecord.dx_label,
                Lesion.split,
                Lesion.external_lesion_id,
            )
            .join(Lesion, ImageRecord.lesion_id == Lesion.id)
            .where(
                ImageRecord.source == "ham10000",
                ImageRecord.dx_label.is_not(None),
                Lesion.split.in_(["train", "val", "test"]),
            )
        ).all()

    if not rows:
        sys.exit("No images found in the database. Run load_ham10000.py first.")

    df = pd.DataFrame([tuple(r) for r in rows], columns=["image_id", "dx", "split", "lesion_id"])

    leaked = (df.groupby("lesion_id")["split"].nunique() > 1).sum()
    if leaked:
        sys.exit(f"LEAKAGE: {leaked} lesions appear in more than one split. Do not train.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df)} rows to {args.out}  (expected 10015 for full HAM10000)")
    print(pd.crosstab(df["dx"], df["split"]).to_string())


if __name__ == "__main__":
    main()