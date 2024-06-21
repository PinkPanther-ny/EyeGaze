import argparse
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from augmentation import train_aug, val_aug
from dataset import GazeDataset
from vit import GazeNet

from torch.cuda.amp import GradScaler, autocast  # 导入混合精度训练的相关模块

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

def gather_and_calculate_mean(val_loss, world_size):
    gathered_losses = [torch.zeros(1).to(DEVICE) for _ in range(world_size)]
    dist.all_gather(gathered_losses, val_loss)
    return sum(gathered_losses).item() / world_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument('-e', "--epoch", default=160, type=int, help="Total epochs")
    parser.add_argument('-l', "--lr", default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument('-n', "--log_name", default="EyeGaze", type=str, help="Current experiment name")
    args = parser.parse_args()

    # Hyperparameters
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epoch

    # DDP backend initialization
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    DEVICE = torch.device("cuda", LOCAL_RANK)
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend='nccl')
    print(f'Using device: {DEVICE}')

    T_0 = 160  # Number of epochs in the first restart cycle (40->120->280->600)
    T_mult = 2  # Multiply the restart cycle length by this factor each restart

    train_dataset = GazeDataset(data_path='images/train', transform=train_aug)
    val_dataset = GazeDataset(data_path='images/val', transform=val_aug)

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)

    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8, shuffle=False)

    # Model, optimizer, and loss function
    model = GazeNet()
    # model.load_state_dict(torch.load('saved_models_pretrain/vit_backbone_freezed_train_head_8v100_116outof120ep_newdata__256.pth'))
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Wrap our model with DDP
    model.to(DEVICE)
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=False)

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

    scaler = GradScaler()  # 初始化GradScaler

    # Training and validation loops
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=LOCAL_RANK != 0)
        for batch_idx, (data, target) in pbar:
            global_step = epoch * len(train_loader) + batch_idx  # Calculate global step
            float_epoch_to_int = int((epoch + batch_idx / len(train_loader)) * 1e3)

            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()

            with autocast():  # 使用autocast进行混合精度训练
                predictions = model(data)
                loss = loss_function(predictions, target)

            scaler.scale(loss).backward()  # 使用scaler.scale
            scaler.step(optimizer)  # 使用scaler.step
            scaler.update()  # 更新scaler

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

        # Validation loop
        if epoch%5 == 0 or epoch > 100:
            model.eval()
            val_loss = torch.zeros(1).to(DEVICE)
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    with autocast():  # 在验证时也使用autocast
                        predictions = model(data)
                        val_loss += loss_function(predictions, target).item()

            val_loss /= len(val_loader)
            mean_val_loss = gather_and_calculate_mean(val_loss, dist.get_world_size())

            if LOCAL_RANK == 0:
                metrics = {
                    'Val/Loss': mean_val_loss,
                    'Val/Pixel dist': calculate_pixel_distance(mean_val_loss),
                }
                for key, value in metrics.items():
                    writer.add_scalar(key, value, epoch)

                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_val_epoch = epoch
                    print(f'\nNew best validation set on epoch {best_val_epoch}, '
                        f'Avg loss: {best_val_loss:.4f} | {calculate_pixel_distance(best_val_loss):.2f} pixels\n')
                    torch.save(model.module.state_dict(), os.path.join(saved_models_dir, f'best.pth'))

                print(f'Validation avg loss: {mean_val_loss:.4f} | {calculate_pixel_distance(mean_val_loss):.2f} pixels')
                print(f'Best validation set on epoch {best_val_epoch}, '
                    f'Avg loss: {best_val_loss:.4f} | {calculate_pixel_distance(best_val_loss):.2f} pixels\n')
                # torch.save(model.module.state_dict(), os.path.join(saved_models_dir, f'epoch_{epoch}.pth'))
