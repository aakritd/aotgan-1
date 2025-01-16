import cv2
import numpy as np
import random

# Define the canvas size
canvas_size = (512, 512)

# Create a black canvas
mask = np.zeros(canvas_size, dtype=np.uint8)

# Generate random points to draw shapes
num_shapes = 4 # Random number of shapes
for _ in range(num_shapes):
    shape_type = random.choice(["circle", "rectangle", "ellipse"])  # Random shape type
    
    if shape_type == "circle":
        center = (random.randint(0, canvas_size[1]), random.randint(0, canvas_size[0]))
        radius = random.randint(10, 100)
        cv2.circle(mask, center, radius, 255, -1)  # Draw filled circle
    
    elif shape_type == "rectangle":
        top_left = (random.randint(0, canvas_size[1]), random.randint(0, canvas_size[0]))
        bottom_right = (random.randint(top_left[0], canvas_size[1]), random.randint(top_left[1], canvas_size[0]))
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)  # Draw filled rectangle
    
    elif shape_type == "ellipse":
        center = (random.randint(0, canvas_size[1]), random.randint(0, canvas_size[0]))
        axes = (random.randint(20, 100), random.randint(20, 100))
        angle = random.randint(0, 360)
        start_angle = 0
        end_angle = 360
        cv2.ellipse(mask, center, axes, angle, start_angle, end_angle, 255, -1)  # Draw filled ellipse

# Save the binary mask
output_path = "random_binary_mask.png"
cv2.imwrite(output_path, mask)

print(f"Binary mask saved as {output_path}")
