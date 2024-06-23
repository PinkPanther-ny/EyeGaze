import os
import numpy as np
import matplotlib.pyplot as plt

# Set the screen resolution
screen_width = 1920
screen_height = 1080

# Directory containing images
image_dir = 'images/train'

# Initialize a list to store the coordinates
coordinates = []

# Loop through the files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        parts = filename.split('_')
        rel_x = float(parts[0])
        rel_y = float(parts[1])
        abs_x = int(rel_x * screen_width)
        abs_y = int(rel_y * screen_height)
        coordinates.append((abs_x, abs_y))

# Convert the list of coordinates to a NumPy array
coordinates = np.array(coordinates)

# Create a heatmap
heatmap, xedges, yedges = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=[screen_width // 10, screen_height // 10])
heatmap = np.rot90(heatmap)
heatmap = np.flipud(heatmap)

# Plot the heatmap
plt.imshow(heatmap, cmap='hot', interpolation='bilinear', extent=[0, screen_width, 0, screen_height])
plt.colorbar(label='Frequency')
plt.title('Heatmap of Gaze Points')
plt.xlabel('Screen Width (pixels)')
plt.ylabel('Screen Height (pixels)')
plt.show()
