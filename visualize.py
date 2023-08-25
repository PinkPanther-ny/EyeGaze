import cv2
import pygame
import torch
from PIL import Image
from torchvision import transforms

from model import GazeNet

device = 'cuda'
# Load the model
model_path = 'saved_models/best.pth'
model = GazeNet().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Open camera 0
cap = cv2.VideoCapture(0)

# Prepare Pygame window
pygame.init()
screen_size = (1920, 1080)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Gaze Position Visualization')

# Image transformation pipeline
transform = transforms.Compose([
    # Resize the short side to 224 while maintaining aspect ratio
    transforms.Resize((224, 299)),  # (224/480) * 640 = 298.667
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Run loop to capture frames and visualize gaze
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        print('Failed to capture frame')
        break

    # Preprocess the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()[0]

    # Map the normalized gaze position to screen coordinates
    gaze_x, gaze_y = (int(prediction[0] * screen_size[0]), int(prediction[1] * screen_size[1]))

    # Update Pygame window
    screen.fill((255, 255, 255))  # Fill background with white
    pygame.draw.circle(screen, (255, 0, 0), (gaze_x, gaze_y), 10)  # Draw red dot at gaze position
    pygame.display.flip()

    # Check for exit event
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
        if event.type == pygame.QUIT:
            running = False

# Clean up
cap.release()
pygame.quit()
