import cv2
import numpy as np
import json
import os

def extract_red_lines(image_path):
    # Load the image
    image = cv2.imread(image_path)
    h, w, c = image.shape

    assert w == 1280 and h == 720, "Image resolution must be 1280x720."
    
    # Convert the image to the RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define the range of red color in RGB
    # These values may need to be adjusted depending on the shade of red
    lower_red = np.array([200, 0, 0])
    upper_red = np.array([255, 100, 100])
    
    # Create a mask to only select red
    mask = cv2.inRange(image_rgb, lower_red, upper_red)
    
    # Find coordinates of red pixels
    red_coords = np.column_stack(np.where(mask > 0))

    import pdb; pdb.set_trace() 
    
    h_samples = [h for h in range(160, 720, 10)]
    

    lanes = []
    for h in h_samples:
        lane = []
        lanes.append(lane)
    
    lanes = np.array(lanes).transpose() # (56, n) - n # of lanes
    lanes = lanes.tolist()

    return

extract_red_lines('test.png')

