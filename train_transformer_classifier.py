"""
Train a Transformer-based image classifier on an RGB dataset.

This script expects the dataset directory to have the following structure::

    dataset/
        train/
            class_a/
                img001.jpg
            class_b/
                img010.jpg
        val/
            class_a/
                img101.jpg
            class_b/
                img110.jpg

The script uses a Vision Transformer (ViT-B/16) from Torchvision and fine-tunes
it on the provided dataset. The transformer encoder enables the model to handle
long-range dependencies across image patches, which is well-suited for
multi-class classification tasks.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models.vision_transformer import ViT_B_16_Weights


@dataclass
class TrainConfig:
    data_dir: Path
    output_dir: Path
    num_classes: int
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class AverageMeter:
    """Utility for tracking running averages."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create training and validation transforms."""

    normalization = ViT_B_16_Weights.IMAGENET1K_V1.transforms().transforms[-1]

    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalization,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalization,
        ]
    )

    return train_transform, val_transform


def create_dataloaders(config: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    train_transform, val_transform = build_transforms(config.image_size)

    train_dataset = datasets.ImageFolder(config.data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(config.data_dir / "val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_model(num_classes: int) -> nn.Module:
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)
    embedding_dim = model.heads.head.in_features
    model.heads.head = nn.Linear(embedding_dim, num_classes)
    return model


def save_checkpoint(state: Dict[str, torch.Tensor], output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / filename
    torch.save(state, checkpoint_path)


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = preds.eq(target).sum().item()
        return correct / target.size(0)


def run_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: CosineAnnealingLR | None = None,
    train: bool = True,
) -> Tuple[float, float]:
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()

    model.train(mode=train)

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_acc = accuracy(outputs, targets)
        epoch_loss.update(loss.item(), images.size(0))
        epoch_acc.update(batch_acc, images.size(0))

    return epoch_loss.avg, epoch_acc.avg


def warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int) -> CosineAnnealingLR:
    return CosineAnnealingLR(optimizer, T_max=warmup_steps, eta_min=optimizer.param_groups[0]["lr"])


def train(config: TrainConfig) -> Dict[str, float]:
    device = torch.device(config.device)
    train_loader, val_loader = create_dataloaders(config)
    model = build_model(config.num_classes).to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config.epochs)

    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(config.to_json())

    global_step = 0

    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")

        train_loss, train_acc = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler=scheduler,
            train=True,
        )

        val_loss, val_acc = run_one_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            scheduler=None,
            train=False,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, config.output_dir, "best.pt")

        save_checkpoint({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }, config.output_dir, "last.pt")

        global_step += len(train_loader)

    metrics_path = config.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2))

    return {"best_val_acc": best_val_acc}


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a Transformer-based image classifier.")
    parser.add_argument("data_dir", type=Path, help="Path to dataset directory with train/ and val/ subdirectories.")
    parser.add_argument("num_classes", type=int, help="Number of classes in the dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to store checkpoints and metrics.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Input image size for the transformer.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay for AdamW optimizer.")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device."
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")

    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    print("Training configuration:\n" + config.to_json())
    metrics = train(config)
    print(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
