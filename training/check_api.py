import argparse
from pathlib import Path

import pandas as pd
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--splits", type=Path, default=Path("../data/splits.csv"))
    ap.add_argument("--data-root", type=Path, default=Path("../data"))
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    health = requests.get(f"{args.url}/health", timeout=10).json()
    print("Health:", health)
    if not health.get("model_loaded"):
        raise SystemExit("API has no model loaded. Register a model and restart the API.")

    df = pd.read_csv(args.splits)
    df = df[df["split"] == "test"].sample(args.n, random_state=0)
    paths = {p.stem: p for p in args.data_root.rglob("*.jpg")}

    correct = 0
    total = 0
    for image_id, dx in zip(df["image_id"], df["dx"]):
        with open(paths[image_id], "rb") as f:
            r = requests.post(
                f"{args.url}/predict",
                files={"file": (f"{image_id}.jpg", f, "image/jpeg")},
                data={"consent": "true"},
                timeout=120,
            )
        if r.status_code != 200:
            print("ERROR", r.status_code, r.text)
            break
        body = r.json()
        pred = body["top_class"]
        conf = body["probs"][pred]
        total += 1
        correct += pred == dx
        print(f"{image_id}  true={dx:6s} predicted={pred:6s} conf={conf:.2f}  {'ok' if pred == dx else 'WRONG'}")

    if total:
        print(f"\nAPI accuracy on {total} test images: {correct / total:.2f}  (standalone check was about 0.77)")


if __name__ == "__main__":
    main()