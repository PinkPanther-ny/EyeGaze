from albumentations import (
    CenterCrop,
    Compose,
    CLAHE,
    HueSaturationValue,
    RandomBrightnessContrast,
    RandomGamma,
    Normalize,
    GaussNoise,
    Resize
)
from albumentations.pytorch import ToTensorV2

ORIGINAL_SIZE = 480, 640
from config import TARGET_SIZE

train_aug = Compose(
    [
        Resize(TARGET_SIZE, int(TARGET_SIZE / ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1])),
        CenterCrop(TARGET_SIZE, TARGET_SIZE),
        GaussNoise(p=0.5),
        RandomBrightnessContrast(),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        CLAHE(p=0.5, clip_limit=2.0),
        Normalize(p=1.0, mean=(0.45610908, 0.45873095, 0.48192639), std=(0.25258335, 0.241994, 0.23515741)),
        ToTensorV2()
    ]
)

val_aug = Compose(
    [
        Resize(TARGET_SIZE, int(TARGET_SIZE / ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1])),
        CenterCrop(TARGET_SIZE, TARGET_SIZE),
        Normalize(p=1.0, mean=(0.45610908, 0.45873095, 0.48192639), std=(0.25258335, 0.241994, 0.23515741)),
        ToTensorV2()
    ]
)
