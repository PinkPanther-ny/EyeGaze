import argparse
import os
from ctypes import windll

import matplotlib.pyplot as plt
import numpy as np


def _get_screen_size():
    try:
        return windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def load_coordinates_from_dir(image_dir: str, screen_width: int, screen_height: int):
    coords = []
    if not os.path.isdir(image_dir):
        print(f"Directory not found: {image_dir}")
        return np.array(coords)

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith('.jpg'):
            continue
        parts = filename.split('_')
        if len(parts) < 3:
            # Unexpected filename pattern
            continue
        try:
            rel_x = float(parts[0])
            rel_y = float(parts[1])
        except ValueError:
            continue
        abs_x = int(rel_x * screen_width)
        abs_y = int(rel_y * screen_height)
        coords.append((abs_x, abs_y))
    return np.array(coords)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a heatmap of gaze points from a directory of labeled images.')
    parser.add_argument('image_dir', help='Directory containing images (e.g., images/train)')
    args = parser.parse_args()

    screen_width, screen_height = _get_screen_size()
    # Directory containing images (from CLI)
    image_dir = os.path.abspath(os.path.expanduser(args.image_dir))

    coordinates = load_coordinates_from_dir(image_dir, screen_width, screen_height)
    if coordinates.size == 0:
        print('No valid coordinates found to plot.')
        raise SystemExit(0)

    # Create a heatmap
    # Use coarser bins for large screens to keep the plot readable
    x_bins = max(10, screen_width // 10)
    y_bins = max(10, screen_height // 10)
    heatmap, xedges, yedges = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=[x_bins, y_bins])
    heatmap = np.rot90(heatmap)
    heatmap = np.flipud(heatmap)

    # Plot the heatmap
    plt.imshow(heatmap, cmap='hot', interpolation='bilinear', extent=(0, screen_width, 0, screen_height))
    plt.colorbar(label='Frequency')
    plt.title('Heatmap of Gaze Points')
    plt.xlabel('Screen Width (pixels)')
    plt.ylabel('Screen Height (pixels)')
    plt.show()
