"""Register a trained model in the database and make it the active one.

Usage:
    python register_model.py --weights models/model.pt --metrics models/metrics.json --name effnet-b0-v1

Then restart the API (it loads the active model at startup).
"""
import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import select, update

from db import SessionLocal
from models import ModelVersion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--name", required=True, help="unique name, e.g. effnet-b0-v1")
    args = ap.parse_args()

    weights = args.weights.resolve()
    if not weights.exists():
        sys.exit(f"Weights file not found: {weights}")
    if not args.metrics.exists():
        sys.exit(f"Metrics file not found: {args.metrics}")

    metrics = json.loads(args.metrics.read_text())
    if "classes" not in metrics:
        sys.exit("metrics.json has no 'classes' list; was it produced by train.py?")

    with SessionLocal() as db:
        if db.scalars(select(ModelVersion).where(ModelVersion.name == args.name)).first():
            sys.exit(f"A model named '{args.name}' already exists. Choose a new name.")

        db.execute(update(ModelVersion).values(is_active=False))  # only one active model
        db.add(
            ModelVersion(
                name=args.name,
                architecture=metrics.get("architecture"),
                metrics=metrics,
                weights_path=str(weights),
                is_active=True,
            )
        )
        db.commit()

    print(f"Registered '{args.name}' as the active model.")
    print(f"Weights: {weights}")
    print("Restart the API to load it.")


if __name__ == "__main__":
    main()