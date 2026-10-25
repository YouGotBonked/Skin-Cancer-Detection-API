"""Train a skin-lesion classifier on the HAM10000 splits exported from the database.

Usage (Colab or local):
    python train.py --splits splits.csv --data-root /folder/containing/the/images --out models

Outputs:
    models/model.pt        TorchScript model, loaded by the API (outputs raw logits)
    models/metrics.json    class order + validation and test metrics

Notes:
  * Preprocessing here (224x224 resize + ImageNet normalization) must match inference.py.
  * The test set is evaluated exactly ONCE, at the end, using the best epoch picked on validation.
  * Educational prototype. Not a medical device.
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose(
    [
        transforms.RandomResizedCrop(SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)
eval_tf = transforms.Compose(
    [transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
)


# ---------------------------------------------------------------- data
def index_images(root: Path) -> dict:
    """Map image_id (filename without extension) -> path, searching root recursively."""
    paths = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths[p.stem] = p
    return paths


class LesionDataset(Dataset):
    def __init__(self, df, paths, class_to_idx, tf):
        self.items = [(paths[i], class_to_idx[d]) for i, d in zip(df["image_id"], df["dx"])]
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        with Image.open(path) as im:
            x = self.tf(im.convert("RGB"))
        return x, y


# ---------------------------------------------------------------- model
def build_model(n_classes: int) -> nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    return m


def set_backbone_trainable(m: nn.Module, flag: bool):
    for p in m.features.parameters():
        p.requires_grad = flag


# ---------------------------------------------------------------- train / eval
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        logits = model(x.to(device))
        P.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def compute_metrics(P, Y, classes):
    P = P.astype(np.float64)
    P = P / P.sum(axis=1, keepdims=True)
    n = len(classes)
    pred = P.argmax(axis=1)
    cm = confusion_matrix(Y, pred, labels=list(range(n)))
    recall = {c: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i, c in enumerate(classes)}
    valid = [v for v in recall.values() if v is not None]
    out = {
        "accuracy": float((pred == Y).mean()),
        "balanced_accuracy": float(np.mean(valid)),
        "per_class_recall": recall,
        "confusion_matrix": cm.tolist(),  # rows = true class, cols = predicted
    }
    try:
        out["macro_auc"] = float(
            roc_auc_score(Y, P, multi_class="ovr", average="macro", labels=list(range(n)))
        )
    except ValueError:
        out["macro_auc"] = None
    if "mel" in classes:
        mel = classes.index("mel")
        y_bin = (Y == mel).astype(int)
        out["mel_auc"] = float(roc_auc_score(y_bin, P[:, mel]))
        fpr, tpr, _ = roc_curve(y_bin, P[:, mel])
        out["mel_sensitivity_at_90_specificity"] = float(tpr[fpr <= 0.10].max())
    return out


def score_for_selection(m):
    return m["macro_auc"] if m["macro_auc"] is not None else m["balanced_accuracy"]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("../models"))
    ap.add_argument("--epochs-head", type=int, default=2, help="epochs with the backbone frozen")
    ap.add_argument("--epochs-ft", type=int, default=25, help="epochs fine-tuning everything")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU found. Training will be very slow. Use Colab with a GPU runtime.")

    # ---- data
    df = pd.read_csv(args.splits)
    classes = sorted(df["dx"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print("Classes:", classes)

    if (df.groupby("lesion_id")["split"].nunique() > 1).any():
        sys.exit("LEAKAGE: a lesion appears in more than one split. Re-export splits.")

    paths = index_images(args.data_root)
    missing = sorted(set(df["image_id"]) - set(paths))
    if missing:
        sys.exit(f"{len(missing)} images from splits.csv not found under {args.data_root}, e.g. {missing[:3]}")

    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    print({s: len(p) for s, p in parts.items()})

    def loader(split, tf, shuffle):
        ds = LesionDataset(parts[split], paths, class_to_idx, tf)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )

    train_loader = loader("train", train_tf, True)
    val_loader = loader("val", eval_tf, False)
    test_loader = loader("test", eval_tf, False)

    # ---- loss with class weights (rarer class -> larger weight, softened by sqrt)
    counts = parts["train"]["dx"].value_counts().reindex(classes).values.astype(np.float64)
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    model = build_model(len(classes)).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"score": -1.0, "state": None, "metrics": None, "epoch": None}

    def run_phase(name, epochs, optimizer, scheduler=None):
        for ep in range(1, epochs + 1):
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
            if scheduler:
                scheduler.step()
            P, Y = predict_probs(model, val_loader, device)
            m = compute_metrics(P, Y, classes)
            score = score_for_selection(m)
            tag = ""
            if score > best["score"]:
                best.update(
                    score=score,
                    state=copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}),
                    metrics=m,
                    epoch=f"{name} {ep}",
                )
                tag = "  <-- best so far"
            print(
                f"[{name} {ep}/{epochs}] loss {loss:.3f} | val macro-AUC {m['macro_auc']:.3f} "
                f"| val balanced-acc {m['balanced_accuracy']:.3f} | {time.time() - t0:.0f}s{tag}"
            )

    # ---- phase 1: train only the new classification head
    if args.epochs_head > 0:
        set_backbone_trainable(model, False)
        opt = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr_head, weight_decay=1e-4)
        run_phase("head", args.epochs_head, opt)

    # ---- phase 2: fine-tune everything at a lower learning rate
    if args.epochs_ft > 0:
        set_backbone_trainable(model, True)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_ft, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_ft)
        run_phase("finetune", args.epochs_ft, opt, sched)

    # ---- final: best epoch on validation -> evaluate on test ONCE
    print(f"\nBest epoch (by validation): {best['epoch']}")
    model.load_state_dict(best["state"])
    P, Y = predict_probs(model, test_loader, device)
    test_metrics = compute_metrics(P, Y, classes)

    print("\n=== TEST RESULTS ===")
    print(f"macro AUC:          {test_metrics['macro_auc']:.3f}")
    print(f"balanced accuracy:  {test_metrics['balanced_accuracy']:.3f}")
    print(f"plain accuracy:     {test_metrics['accuracy']:.3f}  (misleading: 'nv' is ~67% of the data)")
    if "mel_auc" in test_metrics:
        print(f"melanoma AUC:       {test_metrics['mel_auc']:.3f}")
        print(f"melanoma sensitivity at 90% specificity: {test_metrics['mel_sensitivity_at_90_specificity']:.3f}")
    print("per-class recall:")
    for c, r in test_metrics["per_class_recall"].items():
        print(f"  {c:6s} {r:.3f}" if r is not None else f"  {c:6s} n/a")

    # ---- export TorchScript for the API
    args.out.mkdir(parents=True, exist_ok=True)
    model.to("cpu").eval()
    example = torch.randn(1, 3, SIZE, SIZE)
    traced = torch.jit.trace(model, example)
    weights_path = args.out / "model.pt"
    traced.save(str(weights_path))

    reloaded = torch.jit.load(str(weights_path), map_location="cpu").eval()
    with torch.no_grad():
        diff = (reloaded(example) - model(example)).abs().max().item()
    assert diff < 1e-3, f"Exported model differs from trained model (max diff {diff})"

    metrics = {
        "classes": classes,  # the API reads this to label outputs: order matters
        "architecture": "efficientnet_b0",
        "image_size": SIZE,
        "seed": args.seed,
        "best_epoch": best["epoch"],
        "train_counts": {c: int(n) for c, n in zip(classes, counts)},
        "val": best["metrics"],
        "test": test_metrics,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {weights_path} and {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()