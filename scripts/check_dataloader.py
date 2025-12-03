from dataset import get_dataloaders, CLASS_MAPPING

DATA_ROOT = r"D:\Desktop\RTU\Datizrace un zināšanu atklāšana(1),2526-R\2_Lab\wood-defect-detection-2.5"

train_loader, val_loader = get_dataloaders(
    data_root=DATA_ROOT,
    batch_size=2,
    num_workers=0,
    image_size=(512, 512),
)

images, masks = next(iter(train_loader))
print("Images:", images.shape)
print("Masks:", masks.shape)
print("Mask values:", masks.unique())
print("Classes:", CLASS_MAPPING)
