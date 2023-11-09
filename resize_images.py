import cv2
import os

def resize_image(input_path, output_path, width, height):
    # Read the image
    image = cv2.imread(input_path)
    
    # Resize the image
    resized_image = cv2.resize(image, (width, height))
    
    # Save the resized image
    cv2.imwrite(output_path, resized_image)

# Define the target resolution
target_width = 1280
target_height = 720

# Define the input and output directories
input_dir = 'C:/Users/james/Downloads/Speedway'
output_dir = 'C:/Users/james/Downloads/Speedway Resized'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all files in the input directory
files = os.listdir(input_dir)

# Process each file in the input directory
for file_name in files:
    # Check if the file is an image (you may want to add more checks)
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        # Resize the image
        resize_image(input_path, output_path, target_width, target_height)