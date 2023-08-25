import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from dataset import GazeDataset
from model import GazeNet

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = GazeDataset(data_path='images', transform=transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
    ]))

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Hyperparameters
    learning_rate = 0.001
    epochs = 100

    T_0 = 10  # Number of epochs in the first restart cycle
    T_mult = 2  # Multiply the restart cycle length by this factor each restart

    # Model, optimizer, and loss function
    model = GazeNet()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    # Create a learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)

    # Create directory for saving models if it doesn't exist
    saved_models_dir = 'saved_models'
    os.makedirs(saved_models_dir, exist_ok=True)


    best_val_loss = 999999
    best_val_epoch = 0
    # Training and validation loops
    for epoch in range(1, epochs + 1):

        model.train()
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            predictions = model(data)
            loss = loss_function(predictions, target)
            loss.backward()
            optimizer.step()
            # Step the scheduler here
            scheduler.step(epoch - 1 + batch_idx / len(train_loader))

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data),
                    len(train_loader.dataset),
                    100 * batch_idx / len(train_loader),
                    loss.item()
                ))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                predictions = model(data)
                val_loss += loss_function(predictions, target).item()

        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            print(f'\nBest validation set on epoch {best_val_epoch}: Average loss: {best_val_loss:.4f}\n')
            torch.save(model.state_dict(), os.path.join(saved_models_dir, f'best.pth'))
        print(f'\nCur best val loss on epoch {best_val_epoch}: {best_val_loss}, Validation set: Average loss: {val_loss:.4f}\n')
        torch.save(model.state_dict(), os.path.join(saved_models_dir, f'epoch_{epoch}.pth'))
