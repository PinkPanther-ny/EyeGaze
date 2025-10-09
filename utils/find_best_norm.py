import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Directory containing your images
image_dir = 'images'

# Initialize variables to calculate mean and std
mean = np.zeros(3)
std = np.zeros(3)
n_images = 0

# Define a transform to convert images to tensors
transform = transforms.ToTensor()

# Iterate over all images with tqdm
with tqdm(total=len(os.listdir(image_dir)), desc="Calculating mean and std", unit="image") as pbar:
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)

        # Update mean and std
        mean += img_tensor.mean(dim=(1, 2)).numpy()
        std += img_tensor.std(dim=(1, 2)).numpy()
        n_images += 1

        # Update tqdm description
        current_mean = mean / n_images
        current_std = std / n_images
        pbar.set_postfix({
            "mean": [f"{m:.4f}" for m in current_mean],
            "std": [f"{s:.4f}" for s in current_std]
        })
        pbar.update(1)

# Calculate final mean and std
mean /= n_images
std /= n_images

print('Final Mean:', mean)
print('Final Std:', std)
# Mean: [0.50108877 0.47534386 0.47604029]
# Std: [0.25453935 0.25334834 0.25508117]
