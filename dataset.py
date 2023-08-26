import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from augmentation import train_aug, val_aug


def deterministic_shuffle(lst, seed=0):
    random.seed(seed)
    random.shuffle(lst)
    return lst


class GazeDataset(Dataset):
    TRAIN_VAL_SPLIT = 0.9

    def __init__(self, data_path, is_train, transform=None):
        self.data_path = data_path
        self.transform = transform
        # Shuffling the list deterministically
        files = deterministic_shuffle(
            [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))],
            seed=0
        )
        split_pos = int(self.TRAIN_VAL_SPLIT * len(files))
        self.file_list = files[:split_pos] if is_train else files[split_pos:]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        x, y = map(float, file_name.split('_')[:2])
        ground_truth = torch.tensor([x, y], dtype=torch.float)
        img_path = os.path.join(self.data_path, file_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, ground_truth


if __name__ == '__main__':
    train_dataset = GazeDataset(data_path='images', is_train=True, transform=train_aug)
    val_dataset = GazeDataset(data_path='images', is_train=False, transform=val_aug)

    print(len(train_dataset), len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
