# Written using ChatGPT

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import math
import csv
import json
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
from collections import Counter

from model import VisionTransformerEmotion, VisionTransformerEmotionCNN

def accuracy_from_logits(logits, targets):
    """
    logits: (B, num_classes)
    targets: (B,)
    """
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total

def evaluate(model, data_loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples if criterion is not None else None
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def train_one_epoch(model, data_loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    for images, labels in tqdm(data_loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        # Stats
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_samples += batch_size

    epoch_loss = running_loss / running_samples
    epoch_acc = running_correct / running_samples
    return epoch_loss, epoch_acc

def debug_overfit_small_subset(train_loader, device):
    """
    Try to overfit on a very small subset of the data.
    If it can't get near 100% accuracy, something is wrong (data, labels, model).
    """
    from copy import deepcopy

    # Grab a tiny batch
    small_images, small_labels = next(iter(train_loader))
    small_images = small_images.to(device)
    small_labels = small_labels.to(device)

    model = VisionTransformerEmotion().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    print("Debug: overfitting on small subset...")
    for epoch in range(1, 51):  # 50 quick epochs
        model.train()
        optimizer.zero_grad()
        logits = model(small_images)
        loss = criterion(logits, small_labels)
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(dim=1) == small_labels).float().mean().item()
        if epoch % 10 == 0 or acc > 0.99:
            print(f"Epoch {epoch}: loss={loss.item():.4f}, acc={acc:.4f}")
        if acc > 0.99:
            break

    print("Finished small-subset overfit debug.")

def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            return float(current_epoch + 1) / float(max(1, num_warmup_epochs))
        progress = float(current_epoch - num_warmup_epochs) / float(
            max(1, num_training_epochs - num_warmup_epochs)
        )
        # use math.cos, return a Python float
        return 0.5 * (1.0 + math.cos(math.pi * progress))
        # return 0.5 * (1.0 + math.cos(math.pi * (2/3) * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def confusion_matrix_and_per_class_acc(model, data_loader, device, class_names):
    """
    Computes confusion matrix and per-class accuracy.

    class_names: list of class names in the same order as label indices.
    """
    num_classes = len(class_names)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    return cm, per_class_acc

def collect_preds_and_labels(model, data_loader, device):
    """
    Run model over data_loader and collect predictions and true labels.
    Returns:
        all_preds: np.array of shape (N,)
        all_labels: np.array of shape (N,)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_preds, all_labels


def main():
    run_config = {
        "model_type": "vit_cnn",        # "vit_cnn" or "vit_patch"
        "img_size": 48,
        "in_chans": 1,
        "stem_channels": 64,            # if using CNNImageEmbedding
        "patch_size": 4,                # if using patch embedding
        "embed_dim": 192,
        "depth": 6,
        "num_heads": 6,
        "mlp_dim": 768,
        "dropout": 0.2,
        "label_smoothing": 0.05,
        "batch_size": 64,
        "num_epochs": 100,
        "base_lr": 5e-4,
        "weight_decay": 0.01,
        "warmup_epochs": 5,
        "optimizer": "AdamW",
        "scheduler": "cosine_with_warmup",
        "train_augmentations": "Grayscale+Resize+Flip+Rot10+Crop+ColorJitter+Erase",
        "dataset": "emotion-detection-fer",
        "val_split_frac": 0.1,
        "class_weighted_loss": False,
    }

    short_tag = (
        f"{run_config['model_type']}"
        f"_ed{run_config['embed_dim']}"
        f"_d{run_config['depth']}"
        f"_h{run_config['num_heads']}"
        f"_lr{run_config['base_lr']}"
    )

    # Timestamped run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{short_tag}"

    # Base directory for this run
    base_run_dir = os.path.join("runs", run_name)
    os.makedirs(base_run_dir, exist_ok=True)

    print(f"Running experiment: {run_name}")
    print(f"Saving outputs under: {base_run_dir}")

    # Save config.json
    config_path = os.path.join(base_run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Saved config to {config_path}")

    data_root = "~/dl_trans/data/emotion_detection_fer"
    train_dir = os.path.join(data_root, "train")
    test_dir  = os.path.join(data_root, "test")

    # train_transform = transforms.Compose([
    #     transforms.Grayscale(),
    #     transforms.Resize((48, 48)),
        
        # Augmentations
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomRotation(10),       # was 15
        # transforms.RandomResizedCrop(
        #     size=(48, 48),
        #     scale=(0.8, 1.0),       # random zoom, but don’t shrink too tiny
        #     ratio=(0.9, 1.1),
        # ),

    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5]),
    # ])

    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((52, 52)),  # slightly up, so we can crop back to 48

        # Geometric augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [
                transforms.RandomRotation(10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),  # up to 5% shift
                    scale=(0.9, 1.1),        # slight zoom in/out
                ),
            ],
            p=0.7,  # apply this block 70% of the time
        ),

        # Random crop back to 48x48 with some scale jitter
        transforms.RandomResizedCrop(
            size=(48, 48),
            scale=(0.85, 1.0),   # don’t go too tiny
            ratio=(0.9, 1.1),
        ),

        # Photometric augmentations (grayscale but brightness/contrast still matter)
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
        ),

        transforms.ToTensor(),

        # Small random occlusions
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.08),   # small patches
            ratio=(0.3, 3.3),
            value="random",
        ),

        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_transform = val_transform

    full_train_dataset = datasets.ImageFolder(train_dir, transform=None)
    test_dataset       = datasets.ImageFolder(test_dir,  transform=test_transform)

    val_frac = 0.1
    full_len = len(full_train_dataset)
    val_len  = int(full_len * val_frac)
    train_len = full_len - val_len

    train_subset, val_subset = random_split(    
        full_train_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset      # this is a torch.utils.data.Subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            # subset.dataset is the underlying ImageFolder (with transform=None)
            img, label = self.subset.dataset[self.subset.indices[idx]]
            if self.transform is not None:
                img = self.transform(img)
            return img, label
    
    train_dataset = TransformedSubset(train_subset, train_transform)
    val_dataset   = TransformedSubset(val_subset,   val_transform)

    batch_size = run_config["batch_size"]
    num_workers = 4

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))
    print("Class mapping:", full_train_dataset.class_to_idx)

    # class weighted loss
    # Get all labels from the base ImageFolder
    all_labels = np.array(full_train_dataset.targets)  # shape (N,)

    # Restrict to the training split
    train_indices = np.array(train_subset.indices)
    train_labels = all_labels[train_indices]

    label_counts = Counter(train_labels)
    num_classes = len(full_train_dataset.classes)

    # Build a list of counts in index order 0..num_classes-1
    class_counts = [label_counts[i] for i in range(num_classes)]

    print("Train class counts by index:", class_counts)
    print("Class names:", full_train_dataset.classes)

    # Inverse-frequency weights (normalize to mean=1 to keep magnitude reasonable)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.mean()

    print("Class weights:", class_weights.tolist())

    # ==== Model, loss, optimizer, scheduler ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # debug_overfit_small_subset(train_loader, device)

    if run_config["model_type"] == "vit_patch":
        model = VisionTransformerEmotion(
            img_size=run_config["img_size"],
            patch_size=run_config["patch_size"],
            in_chans=run_config["in_chans"],
            num_classes=7,
            embed_dim=run_config["embed_dim"],
            depth=run_config["depth"],
            num_heads=run_config["num_heads"],
            mlp_dim=run_config["mlp_dim"],
            dropout=run_config["dropout"],
        ).to(device)
    elif run_config["model_type"] == "vit_cnn":
        model = VisionTransformerEmotionCNN(
            img_size=run_config["img_size"],
            in_chans=run_config["in_chans"],
            num_classes=7,
            stem_channels=run_config["stem_channels"],
            embed_dim=run_config["embed_dim"],
            depth=run_config["depth"],
            num_heads=run_config["num_heads"],
            mlp_dim=run_config["mlp_dim"],
            dropout=run_config["dropout"],
        ).to(device)

    if run_config["class_weighted_loss"]:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=run_config["label_smoothing"],
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=run_config["label_smoothing"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config["base_lr"],
        weight_decay=run_config["weight_decay"],
    )

    num_epochs = run_config["num_epochs"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_epochs=run_config["warmup_epochs"],
        num_training_epochs=num_epochs,
    )

    # Simple cosine scheduler over epochs
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=num_epochs
    # )

    best_val_acc = 0.0
    best_model_path = os.path.join(base_run_dir, "best_model.pt")

    train_losses_history = []
    val_losses_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, optimizer, criterion
        )

        val_loss, val_acc = evaluate(
            model, val_loader, device, criterion
        )

        scheduler.step()

        train_losses_history.append(train_loss)
        val_losses_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        print(
            f"Train   - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}\n"
            f"Val     - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n"
            f"LR now  - {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"--> New best model saved with val_acc = {best_val_acc:.4f}")

    # Save loss and acc of each epoch
    metrics_path = os.path.join(base_run_dir, "training_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for epoch in range(1, num_epochs + 1):
            writer.writerow([
                epoch,
                train_losses_history[epoch - 1],
                val_losses_history[epoch - 1],
                train_acc_history[epoch - 1],
                val_acc_history[epoch - 1],
            ])
    print(f"Saved metrics to {metrics_path}")
    
    # Save plots of loss and acc over epochs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Loss curves: train + val
    plt.figure()
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses_history, label="Train loss")
    plt.plot(epochs, val_losses_history, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(base_run_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss plot to {loss_plot_path}")

    # Accuracy curves
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_acc_history, label="Train acc")
    plt.plot(range(1, num_epochs + 1), val_acc_history, label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_plot_path = os.path.join(base_run_dir, "accuracy_curve.png")
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Saved accuracy plot to {acc_plot_path}")

    # ==== Evaluate on test set with best model ====
    print("\nTraining complete. Loading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # ----- Classification metrics on test set -----
    class_names = full_train_dataset.classes  # ['angry', 'disgusted', ..., 'surprised']
    num_classes = len(class_names)

    all_preds, all_labels = collect_preds_and_labels(model, test_loader, device)

    # Confusion matrix
    cm_counts = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # Row-normalized confusion matrix (values between 0 and 1)
    cm = cm_counts.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1.0)
    cm /= row_sums

    # Per-class precision, recall, F1
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        average=None,
    )

    # Macro / weighted averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
    )

    print("\nPer-class metrics:")
    for idx, name in enumerate(class_names):
        print(
            f"{idx} ({name}): "
            f"support={supports[idx]}, "
            f"precision={precisions[idx]:.3f}, "
            f"recall={recalls[idx]:.3f}, "
            f"f1={f1s[idx]:.3f}"
        )

    print("\nMacro average: "
          f"precision={macro_p:.3f}, recall={macro_r:.3f}, f1={macro_f1:.3f}")
    print("Weighted average: "
          f"precision={weighted_p:.3f}, recall={weighted_r:.3f}, f1={weighted_f1:.3f}")

    total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", total_params)

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(name, params)

    results_path = os.path.join(base_run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "best_val_acc": best_val_acc,
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "macro_precision": float(macro_p),
                "macro_recall": float(macro_r),
                "macro_f1": float(macro_f1),
                "weighted_precision": float(weighted_p),
                "weighted_recall": float(weighted_r),
                "weighted_f1": float(weighted_f1),
                "total_params": total_params,
            },
            f,
            indent=2,
        )
    print(f"Saved final metrics to {results_path}")

    cm_path = os.path.join(base_run_dir, "confusion_matrix.npy")
    np.save(cm_path, cm)
    print(f"Saved confusion matrix to {cm_path}")

    # Confustion Matrix Plot
    plt.figure(figsize=(6, 6))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")  # <-- Blues cmap
    plt.title("Confusion Matrix (Test)")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Add normalized values inside each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j,
                i,
                f"{val:.2f}",  # show like 0.59, 0.04, etc.
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    cm_plot_path = os.path.join(base_run_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")


if __name__ == "__main__":
    main()

