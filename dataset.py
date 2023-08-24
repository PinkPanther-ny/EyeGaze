import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class GazeDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.file_list = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        x, y = map(float, file_name.split('_')[:2])
        ground_truth = torch.tensor([x, y], dtype=torch.float)
        img_path = os.path.join(self.data_path, file_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, ground_truth


if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image
        transforms.RandomHorizontalFlip(),  # Simple Augmentation
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
    ])

    dataset = GazeDataset(data_path='images', transform=data_transforms)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(len(train_dataset), len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
