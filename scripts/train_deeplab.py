import torch
import torch.nn as nn
from dataset import get_dataloaders, CLASS_MAPPING
from models import create_unet
from metrics import (
    pixel_accuracy,
    intersection_and_union,
    compute_mean_iou,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = r"D:\Desktop\RTU\Datizrace un zināšanu atklāšana(1),2526-R\2_Lab\wood-defect-detection-2.5"

def train_one_epoch(model, loader, optimizer, criterion, num_classes):
    model.train()
    running_loss = 0.0
    running_pa = 0.0
    total_inter = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)
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
        running_pa += pixel_accuracy(logits, masks)

        inter, uni = intersection_and_union(logits, masks, num_classes)
        total_inter += inter
        total_union += uni

    n = len(loader)
    return (
        running_loss / n,
        running_pa / n,
        compute_mean_iou(total_inter, total_union, ignore_background=True),
    )

def validate_one_epoch(model, loader, criterion, num_classes):
    model.eval()
    running_loss = 0.0
    running_pa = 0.0
    total_inter = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)
    total_union = torch.zeros(num_classes, dtype=torch.float64, device=DEVICE)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, masks)

            running_loss += loss.item()
            running_pa += pixel_accuracy(logits, masks)

            inter, uni = intersection_and_union(logits, masks, num_classes)
            total_inter += inter
            total_union += uni

    n = len(loader)
    return (
        running_loss / n,
        running_pa / n,
        compute_mean_iou(total_inter, total_union, ignore_background=True),
    )

def main():
    num_classes = len(CLASS_MAPPING)
    train_loader, val_loader = get_dataloaders(
        data_root=DATA_ROOT,
        batch_size=2,
        num_workers=0,
        image_size=(512, 512),
    )

    model = create_unet("resnet34").to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_miou = 0.0
    save_path = "best_unet_resnet34.pth"

    for epoch in range(1, 21):
        train_loss, train_pa, train_miou = train_one_epoch(
            model, train_loader, optimizer, criterion, num_classes
        )
        val_loss, val_pa, val_miou = validate_one_epoch(
            model, val_loader, criterion, num_classes
        )

        print(
            f"[U-Net] Epoch {epoch:02d} | "
            f"Train: loss={train_loss:.4f}, PA={train_pa:.4f}, mIoU={train_miou:.4f} | "
            f"Val: loss={val_loss:.4f}, PA={val_pa:.4f}, mIoU={val_miou:.4f}"
        )

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved, mIoU={best_val_miou:.4f}")

if __name__ == "__main__":
    main()