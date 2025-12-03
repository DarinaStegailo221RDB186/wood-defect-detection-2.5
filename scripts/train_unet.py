import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_dataloaders, CLASS_MAPPING
from models import create_unet, create_deeplabv3plus
from metrics import pixel_accuracy, intersection_and_union, compute_mean_iou

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_classes: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_pixel_acc = 0.0

    total_intersection = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)
    total_union = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)

    for images, masks in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pa = pixel_accuracy(logits, masks)
        running_pixel_acc += pa

        intersection, union = intersection_and_union(logits, masks, num_classes)
        total_intersection += intersection
        total_union += union

    epoch_loss = running_loss / len(loader)
    epoch_pixel_acc = running_pixel_acc / len(loader)
    epoch_miou = compute_mean_iou(total_intersection, total_union, ignore_background=True)

    return epoch_loss, epoch_pixel_acc, epoch_miou


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_pixel_acc = 0.0

    total_intersection = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)
    total_union = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, masks)

            running_loss += loss.item()
            pa = pixel_accuracy(logits, masks)
            running_pixel_acc += pa

            intersection, union = intersection_and_union(logits, masks, num_classes)
            total_intersection += intersection
            total_union += union

    epoch_loss = running_loss / len(loader)
    epoch_pixel_acc = running_pixel_acc / len(loader)
    epoch_miou = compute_mean_iou(total_intersection, total_union, ignore_background=True)

    return epoch_loss, epoch_pixel_acc, epoch_miou


def main():
    data_root = r"D:\Desktop\RTU\Datizrace un zināšanu atklāšana(1),2526-R\2_Lab\wood-defect-detection-2.5"

    batch_size = 2
    num_epochs = 20
    image_size = (512, 512)
    num_classes = len(CLASS_MAPPING)

    train_loader, val_loader = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=0,
        image_size=image_size,
    )

    model = create_deeplabv3plus(encoder_name="resnet50")

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_miou = 0.0
    save_path = "best_model_deeplabv3plus.pth"

    for epoch in range(1, num_epochs + 1):
        train_loss, train_pa, train_miou = train_one_epoch(
            model, train_loader, optimizer, criterion, num_classes
        )
        val_loss, val_pa, val_miou = validate_one_epoch(
            model, val_loader, criterion, num_classes
        )

        print(
            f"Epoch {epoch:02d}: "
            f"Train loss={train_loss:.4f}, PA={train_pa:.4f}, mIoU={train_miou:.4f} | "
            f"Val loss={val_loss:.4f}, PA={val_pa:.4f}, mIoU={val_miou:.4f}"
        )
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved! mIoU={best_val_miou:.4f}")

    print("Training finished.")
    print(f"Best val mIoU: {best_val_miou:.4f}")


if __name__ == "__main__":
    main()
