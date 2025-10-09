import os

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

TARGET_SIZE = 384

MEAN = (0.45610908, 0.45873095, 0.48192639)
STD = (0.25258335, 0.241994, 0.23515741)

train_aug = v2.Compose([
    v2.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
    v2.ToImage(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
    v2.RandomAutocontrast(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])

val_aug = v2.Compose([
    v2.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])


class GazeDataset(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

        print(f"GazeDataset Init! {self.data_path} {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        x, y = map(float, file_name.split('_')[:2])
        ground_truth = torch.tensor([x, y], dtype=torch.float)
        image = self.transform(Image.open(os.path.join(self.data_path, file_name)).convert("RGB"))
        return image, ground_truth


if __name__ == '__main__':
    train_dataset = GazeDataset(data_path='D:\\Datasets\\images\\train', transform=train_aug)
    val_dataset = GazeDataset(data_path='D:\\Datasets\\images\\val', transform=val_aug)

    print(len(train_dataset), len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print(next(iter(train_loader))[0].shape, next(iter(train_loader))[1].shape)
