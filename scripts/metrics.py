import torch

def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).float()
        return correct.mean().item()

def intersection_and_union(logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = torch.zeros(num_classes, dtype=torch.float64, device=logits.device)
        union = torch.zeros(num_classes, dtype=torch.float64, device=logits.device)

        for cls in range(num_classes):
            pred_cls = preds == cls
            target_cls = targets == cls
            inter = (pred_cls & target_cls).sum().float()
            u = (pred_cls | target_cls).sum().float()
            intersection[cls] += inter
            union[cls] += u

    return intersection, union

def compute_iou_per_class(intersection, union):
    return intersection / (union + 1e-6)

def compute_mean_iou(intersection, union, ignore_background: bool = False):
    iou = compute_iou_per_class(intersection, union)
    if ignore_background and iou.numel() > 1:
        iou = iou[1:]
    return iou.mean().item()
