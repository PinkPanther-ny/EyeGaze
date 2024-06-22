import os
import cv2
import numpy as np
from tqdm import tqdm

def calculate_mean_std(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is not None:
        image = image / 255.0  # Normalize image to [0, 1] range
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1))
        return mean, std
    return None, None

def get_all_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main():
    image_dir = 'images'
    image_paths = get_all_image_paths(image_dir)
    means = []
    stds = []

    overall_mean = np.array([0.0, 0.0, 0.0])
    overall_std = np.array([0.0, 0.0, 0.0])
    total_images = len(image_paths)

    progress_bar = tqdm(image_paths, desc="Processing images", dynamic_ncols=True)

    for i, image_path in enumerate(progress_bar):
        mean, std = calculate_mean_std(image_path)
        if mean is not None and std is not None:
            means.append(mean)
            stds.append(std)

            overall_mean = np.mean(means, axis=0)
            overall_std = np.mean(stds, axis=0)

            progress_bar.set_postfix({
                "Current Mean": overall_mean,
                "Current Std": overall_std
            })

    print(f"Overall Mean: {overall_mean}")
    print(f"Overall Std: {overall_std}")

# Overall Mean: [0.45610908 0.45873095 0.48192639]
# Overall Std: [0.25258335 0.241994   0.23515741]
if __name__ == "__main__":
    main()
