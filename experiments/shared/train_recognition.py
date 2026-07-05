"""Build recognition dataset from class catalogue + train a classifier.

Steps:
1. Read class_catalogue.json (83 SKUs, 442 crops)
2. Organize crops into class_name/crop.jpg folder structure
3. Split train/val
4. Train MobileNetV3 classifier
5. Save model
"""

import json
import shutil
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

PROJECT_ROOT = Path("/home/sanket758/Education/BSBI/Masters-Thesis")
CROP_DIR = PROJECT_ROOT / "Dataset" / "processed_retrieval"
CATALOGUE = PROJECT_ROOT / "experiments/annotation_tool" / "data" / "class_catalogue.json"
RECOG_DIR = PROJECT_ROOT / "Dataset" / "recognition"
MODEL_DIR = PROJECT_ROOT / "models"

VAL_SPLIT = 0.2
SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
IMG_SIZE = 224
LR = 0.001
PATIENCE = 10
MIN_SAMPLES = 3  # Min crops per class to include


def build_dataset():
    """Organize crops into class folders, return class list."""
    catalogue = json.load(open(CATALOGUE))
    classes = catalogue["classes"]

    # Filter: keep only classes with >= MIN_SAMPLES crops
    valid = [c for c in classes if c["n_crops"] >= MIN_SAMPLES]
    print(f"Classes in catalogue: {len(classes)}")
    print(f"Classes with >= {MIN_SAMPLES} crops: {len(valid)}")
    print(f"Total crops: {sum(c['n_crops'] for c in valid)}")

    class_names = [c["class_name"] for c in valid]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # Create dirs
    for split in ("train", "val"):
        for name in class_names:
            (RECOG_DIR / split / name).mkdir(parents=True, exist_ok=True)

    # Copy crops to class folders
    all_samples = []  # (class_name, crop_path)
    for c in valid:
        name = c["class_name"]
        for fname in c["crop_fnames"]:
            src = CROP_DIR / fname
            if src.exists():
                all_samples.append((name, src))

    # Shuffle
    random.seed(SEED)
    random.shuffle(all_samples)

    # Group by class for stratified split
    by_class = {}
    for name, path in all_samples:
        by_class.setdefault(name, []).append(path)

    train_samples = []
    val_samples = []
    for name, paths in by_class.items():
        n_val = max(1, int(len(paths) * VAL_SPLIT))
        val_samples.extend((name, p) for p in paths[:n_val])
        train_samples.extend((name, p) for p in paths[n_val:])

    # Copy files
    for split_name, samples in [("train", train_samples), ("val", val_samples)]:
        for class_name, src_path in samples:
            dst = RECOG_DIR / split_name / class_name / src_path.name
            shutil.copy2(src_path, dst)

    print(f"\nRecognition dataset built at {RECOG_DIR}:")
    print(f"  Train: {len(train_samples)} crops")
    print(f"  Val: {len(val_samples)} crops")
    print(f"  Classes: {len(class_names)}")

    # Save class names
    with open(RECOG_DIR / "class_names.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")

    return class_names


class CropDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = Path(root) / split
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img_path), self.class_to_idx[cls]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def train_model(class_names):
    n_classes = len(class_names)
    print(f"\nTraining classifier: {n_classes} classes")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CropDataset(RECOG_DIR, "train", train_transform)
    val_dataset = CropDataset(RECOG_DIR, "val", val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model: ResNet18 (weights cached locally)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Training
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += inputs.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += inputs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        print(f"  Epoch {epoch:2d}/{EPOCHS}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
            }, MODEL_DIR / "recognition_mobilenetv3.pt")
            print(f"    → New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")

    # Also export TorchScript for deployment
    model.eval()
    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    traced = torch.jit.trace(model, example)
    traced.save(str(MODEL_DIR / "recognition_mobilenetv3_script.pt"))
    print(f"Model exported:")
    print(f"  {MODEL_DIR / 'recognition_mobilenetv3.pt'}")
    print(f"  {MODEL_DIR / 'recognition_mobilenetv3_script.pt'}")
    print(f"  Classes: {n_classes} ({class_names[:5]}...)")
    print(f"  val_acc: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")


def main():
    print("=" * 55)
    print("Recognition Dataset Builder + Classifier")
    print("=" * 55)

    class_names = build_dataset()
    train_model(class_names)


if __name__ == "__main__":
    main()
