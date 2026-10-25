import argparse
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sqlalchemy import select

from db import SessionLocal, engine
from models import Base, ImageRecord, Lesion

SOURCE = "ham10000"
BATCH_SIZE = 500


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def index_images(dirs: list[Path]) -> dict[str, Path]:
    found = {}
    for d in dirs:
        if not d.is_dir():
            sys.exit(f"Images directory not found: {d}")
        for p in d.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                found[p.stem] = p
    return found


def assign_splits(lesions: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> dict[str, str]:
    train_val, test = train_test_split(
        lesions, test_size=test_frac, stratify=lesions["dx"], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_frac / (1 - test_frac),
        stratify=train_val["dx"],
        random_state=seed,
    )
    mapping = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        for lesion_id in frame["lesion_id"]:
            mapping[lesion_id] = name
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, nargs="+", type=Path)
    ap.add_argument("--data-root", type=Path, help="file_path is stored relative to this (default: metadata's folder)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--create-tables", action="store_true", help="create tables if missing (use Alembic later)")
    ap.add_argument("--dry-run", action="store_true", help="compute splits and report, write nothing")
    args = ap.parse_args()

    data_root = (args.data_root or args.metadata.parent).resolve()

    meta = pd.read_csv(args.metadata)
    required = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"}
    missing = required - set(meta.columns)
    if missing:
        sys.exit(f"Metadata is missing columns: {sorted(missing)}")

    paths = index_images(args.images_dir)
    meta["path"] = meta["image_id"].map(paths)
    n_missing = meta["path"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} metadata rows have no image file; they will be skipped.")
        meta = meta.dropna(subset=["path"])

    per_lesion_dx = meta.groupby("lesion_id")["dx"].nunique()
    if (per_lesion_dx > 1).any():
        sys.exit("Some lesions have conflicting dx labels; fix metadata before loading.")

    lesions = meta.groupby("lesion_id", as_index=False).agg(
        dx=("dx", "first"), age=("age", "first"), sex=("sex", "first"), localization=("localization", "first")
    )

    split_of = assign_splits(lesions, args.val_frac, args.test_frac, args.seed)

    print(f"{len(meta)} images across {len(lesions)} lesions")
    print("Lesion-level split:", dict(Counter(split_of.values())))
    print("Image-level split: ", dict(Counter(meta["lesion_id"].map(split_of))))
    print("Class balance per split (images):")
    print(pd.crosstab(meta["dx"], meta["lesion_id"].map(split_of)).to_string())

    if args.dry_run:
        print("\nDry run: nothing written.")
        return

    if args.create_tables:
        Base.metadata.create_all(engine)

    with SessionLocal() as db:
        existing_lesions = {
            row.external_lesion_id: row.id
            for row in db.execute(select(Lesion.external_lesion_id, Lesion.id).where(Lesion.source_dataset == SOURCE))
        }
        existing_hashes = set(db.scalars(select(ImageRecord.sha256)))

        new_lesions = []
        for row in lesions.itertuples(index=False):
            if row.lesion_id in existing_lesions:
                continue
            new_lesions.append(
                Lesion(
                    source_dataset=SOURCE,
                    external_lesion_id=row.lesion_id,
                    split=split_of[row.lesion_id],
                    age=None if pd.isna(row.age) else float(row.age),
                    sex=None if pd.isna(row.sex) else str(row.sex),
                    localization=None if pd.isna(row.localization) else str(row.localization),
                )
            )
        db.add_all(new_lesions)
        db.flush()
        for les in new_lesions:
            existing_lesions[les.external_lesion_id] = les.id
        print(f"\nInserted {len(new_lesions)} new lesions")

        added = dup_hash = 0
        pending = 0
        for row in meta.itertuples(index=False):
            path = Path(row.path)
            digest = sha256_of(path)
            if digest in existing_hashes:
                dup_hash += 1
                continue
            with Image.open(path) as im:
                width, height = im.size
            db.add(
                ImageRecord(
                    lesion_id=existing_lesions[row.lesion_id],
                    external_image_id=row.image_id,
                    file_path=os.path.relpath(path.resolve(), data_root),
                    source=SOURCE,
                    dx_label=row.dx,
                    dx_type=row.dx_type,
                    label_source="dataset",
                    sha256=digest,
                    width=width,
                    height=height,
                )
            )
            existing_hashes.add(digest)
            added += 1
            pending += 1
            if pending >= BATCH_SIZE:
                db.flush()
                pending = 0
                print(f"  ...{added} images", end="\r")

        db.commit()
        print(f"Inserted {added} images, skipped {dup_hash} (already present / duplicate file)")

        print("Done. Splits are stored on lesions; images inherit them via lesion_id.")


if __name__ == "__main__":
    main()