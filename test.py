import cv2
import numpy as np
import pygame

# Load the image
frame = cv2.imread("smoothed_image.png")
framea = frame[:, 0:500]
frame2 = cv2.bitwise_not(framea)

gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Initialize Pygame
pygame.init()

# Set up display
window_width, window_height = gray.shape[1], gray.shape[0]
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Threshold Adjuster")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Button dimensions
button_width, button_height = 50, 30

# Initial threshold values
threshold_min = 180
threshold_max = 255

# Minimum line length for filtering
min_line_length = 50  # Set this to an appropriate value based on your gate size and distance

# Function to draw buttons
def draw_buttons():
    pygame.draw.rect(window, RED, (10, 10, button_width, button_height))  # Decrease min threshold
    pygame.draw.rect(window, GREEN, (70, 10, button_width, button_height))  # Increase min threshold
    pygame.draw.rect(window, RED, (150, 10, button_width, button_height))  # Decrease max threshold
    pygame.draw.rect(window, GREEN, (210, 10, button_width, button_height))  # Increase max threshold

    # Button labels
    font = pygame.font.SysFont(None, 24)
    window.blit(font.render('-', True, BLACK), (30, 15))
    window.blit(font.render('+', True, BLACK), (90, 15))
    window.blit(font.render('-', True, BLACK), (170, 15))
    window.blit(font.render('+', True, BLACK), (230, 15))

# Function to apply threshold and convert image to Pygame surface
def apply_threshold():
    _, threshold = cv2.threshold(gray, threshold_min, threshold_max, cv2.THRESH_BINARY)
    threshold_rgb = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)

    # Find and approximate contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and area
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check line lengths within the approximated contour
        for i in range(len(approx)):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % len(approx)][0]
            if np.linalg.norm(pt1 - pt2) >= min_line_length:
                cv2.line(threshold_rgb, tuple(pt1), tuple(pt2), (0, 255, 0), 2)

    surface = pygame.surfarray.make_surface(threshold_rgb.transpose((1, 0, 2)))
    return surface

# Main loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type is pygame.QUIT:
            running = False
        elif event.type is pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            # Button logic for adjusting thresholds
            if 10 <= mouse_x <= 60 and 10 <= mouse_y <= 40:
                threshold_min = max(0, threshold_min - 5)
            elif 70 <= mouse_x <= 120 and 10 <= mouse_y <= 40:
                threshold_min = min(255, threshold_min + 5)
            elif 150 <= mouse_x <= 200 and 10 <= mouse_y <= 40:
                threshold_max = max(0, threshold_max - 5)
            elif 210 <= mouse_x <= 260 and 10 <= mouse_y <= 40:
                threshold_max = min(255, threshold_max + 5)

    # Update display
    window.fill(WHITE)
    draw_buttons()
    threshold_surface = apply_threshold()
    window.blit(threshold_surface, (0, 50))
    pygame.display.flip()

# Quit Pygame
pygame.quit()