import os
import torch
import torch.nn as nn
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torchvision import transforms
from model.lanenet.loss import FocalLoss, DiscriminativeLoss
import numpy as np
import time
import copy
import matplotlib.pyplot as plt

# Set paths to training, validation, and results data
TRAIN_PATH = "/robodata/public_datasets/TUSimple/train_set/training/train.txt"
VAL_PATH   = "/robodata/public_datasets/TUSimple/train_set/training/val.txt"
SAVE_PATH  = "./log"

EPOCHS = 1             # hyper parameter
BATCH_SIZE = 4          # hyper parameter
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use GPU if available


def create_save_path():
    """
    Create the directory to store results,
    if it does not exist already
    """
    if not os.path.isdir(SAVE_PATH):
        os.makedirs(SAVE_PATH)


# Credit: https://github.com/IrohXu/lanenet-lane-detection-pytorch/tree/main
def compute_loss(net_output, binary_label, instance_label, loss_fn):
    k_binary = 10
    k_instance = 0.3
    k_dist = 1.0

    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = loss_fn(binary_seg_logits, binary_label)

    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    binary_loss = binary_loss * k_binary
    var_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    instance_loss = var_loss + dist_loss
    total_loss = binary_loss + instance_loss
    out = net_output["binary_seg_pred"]

    return total_loss, binary_loss, instance_loss, out


def train_step(model, train_dataloader, loss_fn, optimizer, device):
    # determine dataset size
    ds_size = len(train_dataloader.dataset)

    # set model to train mode (enable dropout layers)
    model.train()

    running_loss = 0.0
    running_loss_b = 0.0
    running_loss_i = 0.0

    # Iterate over input and send data to target device
    for inputs, binarys, instances in train_dataloader:
        inputs = inputs.type(torch.FloatTensor).to(device)            # Raw data
        binarys = binarys.type(torch.LongTensor).to(device)           # Binary masks 
        instances = instances.type(torch.FloatTensor).to(device)      # Instances

        # clear optimizer gradients from previous iteration
        optimizer.zero_grad()

        # forward pass - Enable gradient tracking for optimization purposes
        with torch.set_grad_enabled(True):
            outputs = model(inputs)                                     # Make predictions
            loss = compute_loss(outputs, binarys, instances, loss_fn)   # Compute prediction loss

            loss[0].backward()    # back propagate and compute gradients dL/dX
            optimizer.step()      # make the optimizer "step" in direction of minimizing loss

        # update loss 
        running_loss += loss[0].item() * inputs.size(0)
        running_loss_b += loss[1].item() * inputs.size(0)
        running_loss_i += loss[2].item() * inputs.size(0)

    epoch_loss = running_loss / ds_size
    binary_loss = running_loss_b / ds_size
    instance_loss = running_loss_i / ds_size

    return epoch_loss, binary_loss, instance_loss


def val_step(model, val_dataloader, loss_fn, device=DEVICE):
    # determine dataset size
    ds_size = len(val_dataloader.dataset)

    # set model to evaluation mode
    model.eval()

    running_loss = 0.0
    running_loss_b = 0.0
    running_loss_i = 0.0

    # Iterate over input and send data to target device
    for inputs, binarys, instances in val_dataloader:
        inputs = inputs.type(torch.FloatTensor).to(device)
        binarys = binarys.type(torch.LongTensor).to(device)
        instances = instances.type(torch.FloatTensor).to(device)

        # Disable gradient tracking since we are evaluating, not optimizing
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = compute_loss(outputs, binarys, instances, loss_fn)

        # update loss
        running_loss += loss[0].item() * inputs.size(0)
        running_loss_b += loss[1].item() * inputs.size(0)
        running_loss_i += loss[2].item() * inputs.size(0)

    epoch_loss = running_loss / ds_size
    binary_loss = running_loss_b / ds_size
    instance_loss = running_loss_i / ds_size

    return epoch_loss, binary_loss, instance_loss


def train_model():
    create_save_path()

    # Set random seed to produce repeatable results
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Data augmentation for training and validation set
    train_tf = transforms.Compose([
            transforms.Resize((512, 256)),                                                  # need to be 512 x 256 for the model
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # color augmentation  
            transforms.ToTensor(),                                                          # convert array to Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])            # normalize color scale

    val_tf = transforms.Compose([
            transforms.Resize((512, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Since data is rescaled, the labels / masks (i.e. targets) should also be identically manipulated
    target_tf = transforms.Compose([Rescale((512, 256))])

    # Build Datasets
    train_dataset = TusimpleSet(TRAIN_PATH, transform=train_tf, target_transform=target_tf)
    val_dataset = TusimpleSet(VAL_PATH, transform=val_tf, target_transform=target_tf)

    # Build Dataloaders
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define model and send to target device (GPU OR CPU)
    model = LaneNet(arch="DeepLabv3+")
    model.to(DEVICE)
    model.train()       # set to train model to enable dropouts, batch norm, etc

    # Define loss function to compute training and validation loss
    # Default recommended for model: FocalLoss
    loss_fn = FocalLoss(gamma=2, alpha=[0.25, 0.75])

    # Optimizer to minimize loss - extension of Gradient Descent
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Store the results of each training epoch
    results = {'train_loss': [], 'val_loss_total': [], 'val_loss_binary': [], 'val_loss_instance': []}

    print(f"Beginning training for LaneNet for {EPOCHS} epochs...\n")

    # Keep track of best performing model weights (i.e. minimum loss) during training
    best_loss = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = None

    start_time = time.time()

    for epoch in range(1, EPOCHS+1):
        print(f'Epoch: {epoch}\n-------')

        train_losses = train_step(model, train_dl, loss_fn, optimizer, DEVICE)
        print(f"TRA Total Loss: {train_losses[0]:.4f} Binary Loss: {train_losses[1]:.4f} Instance Loss: "
              f"{train_losses[2]:.4f}")

        val_losses = val_step(model, val_dl, loss_fn, DEVICE)
        print(f"VAL Total Loss: {val_losses[0]:.4f} Binary Loss: {val_losses[1]:.4f} Instance Loss: "
              f"{val_losses[2]:.4f}")

        print(f"\n{time.time() - start_time:.1f} second since start of training\n")

        results['train_loss'].append(train_losses[0])
        results['val_loss_total'].append(val_losses[0])
        results['val_loss_binary'].append(val_losses[1])
        results['val_loss_instance'].append(val_losses[2])

        # Update best performing weights, if required
        if val_losses[0] <= best_loss:
            best_loss = val_losses[0]
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    end_time = time.time()
    print(f"Toal Execution time on {DEVICE}: {format(end_time - start_time, '0.1f')} seconds\n")

    print(f"Best epoch: {best_epoch}")
    print(f'Minimum validation loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    model_filename = os.path.join(SAVE_PATH, 'LaneNet_Best.pth')
    torch.save(model.state_dict(), model_filename)                       # Save Model
    print("model is saved: {}".format(model_filename))

    # Save plots for train validation loss
    plt.plot([e for e in range(1, EPOCHS+1)], results['train_loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Training Loss")
    plt.savefig(f"{SAVE_PATH}/train_loss.png")
    plt.close()

    plt.plot([e for e in range(1, EPOCHS+1)], results['val_loss_total'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Validation Loss")
    plt.savefig(f"{SAVE_PATH}/val_loss_total.png")
    plt.close()

    plt.plot([e for e in range(1, EPOCHS+1)], results['val_loss_binary'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Binary Validation Loss")
    plt.savefig(f"{SAVE_PATH}/val_loss_binary.png")
    plt.close()

    plt.plot([e for e in range(1, EPOCHS+1)], results['val_loss_instance'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Instance Validation Loss")
    plt.savefig(f"{SAVE_PATH}/val_loss_instance.png")
    plt.close()


if __name__ == "__main__":
    train_model()
