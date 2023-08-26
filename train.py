import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from dataset import GazeDataset
from model import GazeNet


def calculate_pixel_distance(avg_mse_loss):
    # Convert average MSE loss to Euclidean distance in normalized coordinates
    # Multiplying by 2 because the MSE is averaged over both x and y dimensions
    normalized_distance = math.sqrt(avg_mse_loss * 2)

    # Convert normalized distances to screen coordinates (1920x1080)
    avg_distance_x_pixels = normalized_distance * 1920
    avg_distance_y_pixels = normalized_distance * 1080

    # Calculate the average Euclidean distance in pixels
    avg_pixel_distance = math.sqrt(avg_distance_x_pixels ** 2 + avg_distance_y_pixels ** 2)

    return avg_pixel_distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument('-e', "--epoch", default=150, type=int, help="Total epochs")
    parser.add_argument('-l', "--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument('-n', "--log_name", default="EyeGaze", type=str, help="Current experiment name")
    args = parser.parse_args()

    # Hyperparameters
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epoch

    # DDP backend initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1, 2]))
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    DEVICE = torch.device("cuda", LOCAL_RANK)
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend='nccl')
    print(f'Using device: {DEVICE}')

    T_0 = 25  # Number of epochs in the first restart cycle (25->75->175->375)
    T_mult = 2  # Multiply the restart cycle length by this factor each restart

    dataset = GazeDataset(data_path='images', transform=transforms.Compose([
        # Resize the short side to 224 while maintaining aspect ratio
        transforms.Resize((224, 299)),  # (224/480) * 640 = 298.667
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Model, optimizer, and loss function
    model = GazeNet()

    # Wrap our model with DDP
    model.to(DEVICE)
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    # Create a learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)

    # Create directory for saving models if it doesn't exist
    saved_models_dir = 'saved_models'
    os.makedirs(saved_models_dir, exist_ok=True)

    best_val_loss = 999999
    best_val_epoch = 0

    writer = None
    if LOCAL_RANK == 0:
        # Initialize the TensorBoard writer
        writer = SummaryWriter(log_dir=f'log/{args.log_name}')

    # Training and validation loops
    for epoch in range(epochs):

        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=LOCAL_RANK != 0)
        for batch_idx, (data, target) in pbar:
            global_step = epoch * len(train_loader) + batch_idx  # Calculate global step
            float_epoch_to_int = int((epoch + batch_idx / len(train_loader)) * 1e3)

            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(data)
            loss = loss_function(predictions, target)
            loss.backward()

            optimizer.step()
            scheduler.step(epoch + batch_idx / len(train_loader))

            if LOCAL_RANK == 0:
                metrics = {
                    'Train/Loss': loss.item(),
                    'Train/Pixel dist': calculate_pixel_distance(loss.item()),
                    'Train/Learning Rate': scheduler.get_last_lr()[0]
                }

                # Log to TensorBoard
                for key, value in metrics.items():
                    writer.add_scalar(key, value, global_step)

                # Log to TensorBoard
                for key, value in metrics.items():
                    writer.add_scalar(key + "_norm_epoch", value, float_epoch_to_int)

                # Update progress bar
                pbar.set_description(f'Epoch: {epoch}')
                pbar.set_postfix({key.lower().replace(' ', '_'): f'{value:.6f}' for key, value in metrics.items()})

        if LOCAL_RANK == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    predictions = model(data)
                    val_loss += loss_function(predictions, target).item()

            val_loss /= len(val_loader)
            metrics = {
                'Val/Loss': val_loss,
                'Val/Pixel dist': calculate_pixel_distance(val_loss),
            }
            for key, value in metrics.items():
                writer.add_scalar(key, value, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch
                print(f'\nNew best validation set on epoch {best_val_epoch}, '
                      f'Avg loss: {best_val_loss:.4f} | {calculate_pixel_distance(best_val_loss):.2f} pixels\n')
                torch.save(model.module.state_dict(), os.path.join(saved_models_dir, f'best.pth'))

            print(f'Validation avg loss: {val_loss:.4f} | {calculate_pixel_distance(best_val_loss):.2f} pixels')
            print(f'Best validation set on epoch {best_val_epoch}, '
                  f'Avg loss: {best_val_loss:.4f} | {calculate_pixel_distance(best_val_loss):.2f} pixels\n')
            torch.save(model.module.state_dict(), os.path.join(saved_models_dir, f'epoch_{epoch}.pth'))
