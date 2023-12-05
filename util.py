import cv2
import numpy as np
import json
import os

''' Prepare Dataset'''
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

    # Process each file in the input directory
    for file_name in files:
        # Check if the file is an image (you may want to add more checks)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            
            # Resize the image
            resize_image(input_path, output_path, target_width, target_height)

def get_unique_lane_idx(index):
    lst = index.tolist()
    unique_idx = []
    for i in range(len(lst)):
        if i == len(lst)-1:
            unique_idx.append(lst[i]) 
        elif lst[i] != (lst[i+1]-1):
            unique_idx.append(lst[i]) 
    return unique_idx

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
    # red_coords = np.column_stack(np.where(mask > 0))

    h_samples = [h for h in range(160, 720, 10)]
    max_n_of_lanes = [len(get_unique_lane_idx(mask[h].nonzero()[0])) for h in range(160, 720, 10)]
    max_n_of_lanes = max(max_n_of_lanes)

    lanes = []
    for h in h_samples:
        x_coord = mask[h]
        index = x_coord.nonzero()[0]
        lane = get_unique_lane_idx(index)
        if len(lane) != max_n_of_lanes:
            lane = [-2] * max_n_of_lanes
        lanes.append(lane)
    
    lanes = np.array(lanes).transpose() # (56, n) - n # of lanes
    lanes = lanes.tolist()

    return lanes, h_samples

def generate_label_json():
    with open("label_data_test_custom_test.json", "a") as outfile:
        cases = ["rural", "speedway"]
        n_images = [22, 20]
        for n, case in enumerate(cases):
            indir = "/home/jihwan98/lanenet-lane-detection-pytorch"
            
            for i in range(1, n_images[n]):

                img_path = f'{indir}/{case}_annotated/{case}{i}.png'
                lanes, h_samples = extract_red_lines(img_path)
                data = {
                    "lane": lanes,
                    "h_samples": h_samples,
                    "raw_file": img_path
                }
                data_object = json.dumps(data, separators=(',', ':'))
                outfile.write(data_object)
                outfile.write("\n")


''' Test'''
# overlay detected lines (binary image) on top of input image
def overlay_images(binaryImage, inputImage, saveFile):

    binary = cv2.imread(binaryImage, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(inputImage)

    _, mask = cv2.threshold(binary, 240, 255, cv2.THRESH_BINARY)
    
    
    colorMask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    red = np.array([0, 0, 255], dtype=np.uint8)
    colorMask = cv2.bitwise_and(colorMask, red, mask=mask)

    binColor = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    onlyWhite = cv2.bitwise_and(binColor, colorMask)

    result = cv2.addWeighted(color, 1, onlyWhite, 1, 0)
    cv2.imwrite(saveFile, result)


if __name__ == "__main__":
    generate_label_json()
    input_path = "/home/jihwan98/lanenet-lane-detection-pytorch/test/urban_dark.png"
    # output_path = "./test"
    output_path = "/robodata/public_datasets/TUSimple/train_set/training"
    resize_image(input_path, output_path, 1280, 720) 
    # overlay_images('binary_output.png', 'input.png', 'megusta_comer_pollo.png')
