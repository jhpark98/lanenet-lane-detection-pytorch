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





        
    # # Group red_coords based on their y-coordinates to identify distinct lanes
    # # This is a simplistic grouping approach, you may need a more advanced method
    # # to accurately separate lanes
    # lanes = [[] for _ in range(4)]  # Assume there are 4 lanes

    # # Here, we're assuming that the y-coordinates are horizontal lines
    # # for simplicity, you will need to adjust this if lines are not horizontal.
    # for y, x in red_coords:
    #     lane_index = x // (1280 // 4)  # Rough division of lanes
    #     lanes[lane_index].append(x)
    #     if y not in h_samples:
    #         h_samples.append(y)

    # # Make sure each lane has the same number of width values
    # for lane in lanes:
    #     lane.sort()
    #     lane.extend([None] * (len(h_samples) - len(lane)))

    # return lanes, h_samples

# # Sample usage
# test_dir = 'test_custom'
# image_path = [f'{test_dir}/{i}.png' for i in range(1, 21)]

# # Generate test label.json
# for path in image_path:
#     lanes, h_samples = extract_red_lines(image_path)

#     # Create the JSON structure
#     json_data = {
#         'lanes': lanes,
#         'h_samples': h_samples,
#         'raw_file': path
#     }

#     # Save to JSON file
#     output_path = 'label_data_test_custom.json'
#     with open(output_path, 'w') as json_file:
#         json.dump(json_data, json_file, indent=4)

#     print(f"Lane data has been saved to {output_path}")
