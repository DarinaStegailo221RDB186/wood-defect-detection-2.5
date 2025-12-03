import segmentation_models_pytorch as smp
from dataset import CLASS_MAPPING

NUM_CLASSES = len(CLASS_MAPPING)

def create_unet(encoder_name: str = "resnet34"):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )

def create_deeplabv3plus(encoder_name: str = "resnet50"):
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
