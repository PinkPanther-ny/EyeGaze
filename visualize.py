import random
import time
import cv2
import keyboard
import numpy as np
import pyautogui
import pygame
import torch
from scipy.stats import multivariate_normal

# Image transformation pipeline
from augmentation import val_aug as transform
from vit import GazeNet

device = 'cuda'
# Load the model
# model_path = 'saved_models_hist/vit_backbone_freezed_train_head_8v100_116outof120ep_newdata__256.pth'
model_path = 'saved_models_hist/122_new.pth'
model = GazeNet().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Open camera 0
cap = cv2.VideoCapture(0)

# Prepare Pygame window
pygame.init()
screen_size = (1920, 1080)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Gaze Position Heatmap')

buffer_size = 6  # Keep the last n gaze positions
gaze_buffer = []
gaussian_render_radius = 160
multivariate_covariance = 1500

# Initialize an array for the heat map, flipped to (height, width)
heat_map_array = np.zeros((screen_size[1], screen_size[0]))
colormap_list = [eval(f'cv2.{i}') for i in dir(cv2) if i.startswith('COLORMAP')]
current_colormap = random.choice(colormap_list)

inference_times = []  # To store the last 10 inference times

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        print('Failed to capture frame')
        break

    start_time = time.time()  # Start timer

    # Preprocess the frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image=image)['image'].unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()[0]

    end_time = time.time()  # End timer
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Append inference time to the list and trim if necessary
    inference_times.append(inference_time)
    if len(inference_times) > 10:
        inference_times.pop(0)

    # Calculate average inference time
    avg_inference_time = sum(inference_times) / len(inference_times)

    # Map the normalized gaze position to screen coordinates
    gaze_x, gaze_y = (int(prediction[0] * screen_size[0]), int(prediction[1] * screen_size[1]))

    # Append gaze position to buffer and trim if necessary
    gaze_buffer.append((gaze_x, gaze_y))
    if len(gaze_buffer) > buffer_size:
        gaze_buffer.pop(0)

    # Reset the heat_map_array
    heat_map_array.fill(0)

    # Generate heat map by overlaying Gaussian
    for gx, gy in gaze_buffer:
        x_min, x_max = max(0, gx - gaussian_render_radius), min(screen_size[0], gx + gaussian_render_radius)
        y_min, y_max = max(0, gy - gaussian_render_radius), min(screen_size[1], gy + gaussian_render_radius)

        # Check for valid dimensions
        if x_min >= x_max or y_min >= y_max:
            continue

        x, y = np.mgrid[x_min:x_max, y_min:y_max]

        pos = np.dstack((x, y))
        rv = multivariate_normal([gx, gy], [[multivariate_covariance, 0], [0, multivariate_covariance]])

        clipped_heatmap = rv.pdf(pos)
        heat_map_array[y_min:y_max, x_min:x_max] += clipped_heatmap.T  # Transposed to match shape

    # Normalize heat_map_array for visualization
    max_value = heat_map_array.max()
    if max_value > 0:  # or np.isfinite(max_value) if you want to check for NaN as well
        heat_map_normalized = (heat_map_array / max_value * 255).astype(np.uint8)
    else:
        heat_map_normalized = np.zeros_like(heat_map_array).astype(np.uint8)

    heat_map_colored = cv2.applyColorMap(heat_map_normalized, current_colormap)

    # Create a Pygame surface from the heatmap and display
    heat_map_surface = pygame.surfarray.make_surface(np.transpose(heat_map_colored, (1, 0, 2)))
    screen.blit(heat_map_surface, (0, 0))

    # Display average inference time on the screen
    font = pygame.font.SysFont('Arial', 30)
    text_surface = font.render(f'Avg Inference Time: {avg_inference_time:.2f} ms', True, (255, 255, 255))
    screen.blit(text_surface, (10, 10))

    pygame.display.flip()

    if keyboard.is_pressed('LEFT ALT'):
        pyautogui.moveTo(*np.mean(np.array(gaze_buffer), axis=0), duration=0.1)

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            current_colormap = random.choice(colormap_list)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
        if event.type == pygame.QUIT:
            running = False

cap.release()
pygame.quit()
