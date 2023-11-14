import time
import os
import torch
from model.lanenet.LaneNet import LaneNet
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use GPU if available
TEST_PATH = "./Test_Outputs"                                             # path to store test results
WEIGHTS_PATH = "log/best_model_DeepLab.pth"                                    # path to saved weights

IMG_PATH = "/home/jihwan98/lanenet-lane-detection-pytorch/test/urban_1_resize.png"                                            # Path to image to test on


def run_test():
    # Define model and send to target device and load best performing weights
    model = LaneNet(arch="DeepLabv3+")
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.to(DEVICE)

    # Set to evaluation mode; disables dropout layers, batch norms, etc
    model.eval()

    # Transforms required for all inputs to model
    test_transform = transforms.Compose([
        transforms.Resize((512, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Prepare input image
    input_img = Image.open(IMG_PATH)                  # open
    input_img = test_transform(input_img).to(DEVICE)  # augmentations
    input_img = torch.unsqueeze(input_img, dim=0)     # resize to include batch_size (B, C, H, W)

    # Make prediction(s)
    out = model(input_img)
    instance_pred = torch.squeeze(out['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    binary_pred = torch.squeeze(out['binary_seg_pred']).to('cpu').numpy() * 255

    # Store results
    img_raw = Image.open(IMG_PATH)
    img_raw = img_raw.resize((256, 512))
    img_raw = np.array(img_raw)

    t = time.time()

    cv2.imwrite(os.path.join(TEST_PATH, f'raw_input_image_{t}.jpg'), img_raw)
    cv2.imwrite(os.path.join(TEST_PATH, f'instance_output_{t}.jpg'), instance_pred.transpose((1, 2, 0)))
    cv2.imwrite(os.path.join(TEST_PATH, f'binary_output_{t}.jpg'), binary_pred)

    print("Successfully tested image.")


if __name__ == "__main__":
    if not os.path.exists(TEST_PATH):
        os.mkdir(TEST_PATH)

    run_test()
