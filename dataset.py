import cv2
import os

import torch
from torch.utils.data import Dataset, DataLoader

from augmentation import train_aug, val_aug


class GazeDataset(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        # Read the list of files
        self.file_list = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        
        print(f"GazeDataset Init! {self.data_path} {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        x, y = map(float, file_name.split('_')[:2])
        ground_truth = torch.tensor([x, y], dtype=torch.float)
        img_path = os.path.join(self.data_path, file_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image=image)['image']

        return image, ground_truth


if __name__ == '__main__':
    train_dataset = GazeDataset(data_path='images/train', transform=train_aug)
    val_dataset = GazeDataset(data_path='images/val', transform=val_aug)

    print(len(train_dataset), len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
