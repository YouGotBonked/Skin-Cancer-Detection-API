import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

from inference import Classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("../models/model.pt"))
    ap.add_argument("--metrics", type=Path, default=Path("../models/metrics.json"))
    ap.add_argument("--splits", type=Path, default=Path("../data/splits.csv"))
    ap.add_argument("--data-root", type=Path, default=Path("../data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300, help="sample size (0 = all)")
    args = ap.parse_args()

    classes = json.loads(args.metrics.read_text())["classes"]
    print("Class order used:", classes)
    clf = Classifier(str(args.weights), classes)

    df = pd.read_csv(args.splits)
    df = df[df["split"] == args.split]
    if args.n and args.n < len(df):
        df = df.sample(args.n, random_state=0)

    paths = {p.stem: p for p in args.data_root.rglob("*.jpg")}

    correct = skipped = 0
    true_counts, hit_counts, pred_counts = Counter(), Counter(), Counter()
    examples = []

    for image_id, dx in zip(df["image_id"], df["dx"]):
        if image_id not in paths:
            skipped += 1
            continue
        with Image.open(paths[image_id]) as im:
            probs = clf.predict(im.convert("RGB"))
        pred = max(probs, key=probs.get)
        true_counts[dx] += 1
        pred_counts[pred] += 1
        if pred == dx:
            correct += 1
            hit_counts[dx] += 1
        if len(examples) < 8:
            examples.append((image_id, dx, pred, round(probs[pred], 2)))

    n = sum(true_counts.values())
    if n == 0:
        raise SystemExit("No images were checked. Is --data-root correct?")

    print(f"\nChecked {n} images from the '{args.split}' split ({skipped} skipped, not found)")
    print(f"Accuracy: {correct / n:.3f}   (expected around 0.80)")
    print("\nPer-class recall (correct / total):")
    for c in classes:
        t = true_counts[c]
        print(f"  {c:6s} {hit_counts[c]:4d} / {t:4d}" + (f"  = {hit_counts[c] / t:.2f}" if t else ""))
    print("\nWhat the model predicted overall:", dict(pred_counts))
    print("\nFirst few examples (image, true, predicted, confidence):")
    for e in examples:
        print("  ", e)


if __name__ == "__main__":
    main()