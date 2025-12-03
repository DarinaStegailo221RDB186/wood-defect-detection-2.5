import os
import json
import base64
import zlib
from typing import Callable, Optional, Tuple, Dict, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False


CLASS_MAPPING: Dict[str, int] = {
    "background": 0,
    "Crack": 1,
    "Death_know": 2,
    "Knot_missing": 3,
    "Live_knot": 4,
    "Marrow": 5,
    "Quartzity": 6,
    "knot_with_crack": 7,
    "resin": 8,
}

DEFECT_CLASSES: List[str] = [
    "Crack",
    "Death_know",
    "Knot_missing",
    "Live_knot",
    "Marrow",
    "Quartzity",
    "knot_with_crack",
    "resin",
]

def decode_bitmap_to_array(bitmap_dict: dict) -> np.ndarray:
    raw = base64.b64decode(bitmap_dict["data"])
    decomp = zlib.decompress(raw)
    with Image.open(io.BytesIO(decomp)) as im:
        mask = np.array(im)
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)
    return mask


import io


def build_mask_from_json(
    json_path: str,
    class_mapping: Dict[str, int] = CLASS_MAPPING,
    image_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data["size"]["height"]
    w = data["size"]["width"]

    mask = np.zeros((h, w), dtype=np.uint8)

    for obj in data.get("objects", []):
        class_title = obj["classTitle"]
        if class_title not in class_mapping:
            continue

        cls_id = class_mapping[class_title]

        geom_type = obj.get("geometryType")

        if geom_type == "bitmap":
            bmp = obj["bitmap"]
            patch = decode_bitmap_to_array(bmp)
            ph, pw = patch.shape[:2]
            x0, y0 = bmp["origin"]

            x1 = min(x0 + pw, w)
            y1 = min(y0 + ph, h)

            patch_h = y1 - y0
            patch_w = x1 - x0
            if patch_h <= 0 or patch_w <= 0:
                continue

            patch = patch[:patch_h, :patch_w]

            region = mask[y0:y1, x0:x1]
            region[patch == 1] = cls_id

        elif geom_type == "rectangle":
            points = obj["points"]["exterior"]
            (x0, y0), (x1, y1) = points
            x0 = max(int(x0), 0)
            y0 = max(int(y0), 0)
            x1 = min(int(x1), w)
            y1 = min(int(y1), h)
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = cls_id

        else:
            continue

    return mask


class WoodDefectsDataset(Dataset):

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        use_albumentations: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
    ):

        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.root = root
        self.split = split
        self.use_albumentations = use_albumentations and _HAS_ALBUMENTATIONS
        self.image_size = image_size

        self.img_dir = os.path.join(root, split, "img")
        self.ann_dir = os.path.join(root, split, "ann")

        self.img_paths = []
        self.ann_paths = []

        for fname in sorted(os.listdir(self.img_dir)):
            if not fname.lower().endswith(".jpg"):
                continue
            if fname.startswith("._"):
                continue

            img_path = os.path.join(self.img_dir, fname)
            ann_name = fname + ".json"
            ann_path = os.path.join(self.ann_dir, ann_name)
            if not os.path.exists(ann_path):
                raise FileNotFoundError(f"Annotation not found for {img_path}: {ann_path}")

            self.img_paths.append(img_path)
            self.ann_paths.append(ann_path)

        assert len(self.img_paths) > 0, f"No images found in {self.img_dir}"

        if transforms is None:
            self.transforms = self._default_transforms()
        else:
            self.transforms = transforms

    def _default_transforms(self):
        if self.use_albumentations:
            h, w = (self.image_size if self.image_size is not None else (1024, 2800))
            return A.Compose(
                [
                    A.Resize(height=h, width=w),
                    A.HorizontalFlip(p=0.5 if self.split == "train" else 0.0),
                    A.RandomBrightnessContrast(p=0.5 if self.split == "train" else 0.0),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            def _simple_transform(image: np.ndarray, mask: np.ndarray):
                img = Image.fromarray(image)
                msk = Image.fromarray(mask)

                if self.image_size is not None:
                    img = img.resize(self.image_size[::-1], resample=Image.BILINEAR)
                    msk = msk.resize(self.image_size[::-1], resample=Image.NEAREST)

                img = np.array(img)
                msk = np.array(msk)

                img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                msk = torch.from_numpy(msk).long()
                return {"image": img, "mask": msk}

            return _simple_transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            image = np.array(img)

        mask = build_mask_from_json(
            ann_path,
            class_mapping=CLASS_MAPPING,
            image_size=self.image_size,
        )

        if self.use_albumentations:
            transformed = self.transforms(image=image, mask=mask)
            image_t = transformed["image"].float()
            mask_t = transformed["mask"].long()
        else:
            transformed = self.transforms(image, mask)
            image_t = transformed["image"]
            mask_t = transformed["mask"]

        return image_t, mask_t


def get_dataloaders(
    data_root: str,
    batch_size: int = 2,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = WoodDefectsDataset(
        root=data_root,
        split="train",
        image_size=image_size,
        use_albumentations=True,
    )

    val_dataset = WoodDefectsDataset(
        root=data_root,
        split="val",
        image_size=image_size,
        use_albumentations=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
