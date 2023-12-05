# Lane Detection with Python using PyTorch

## Credits
The dataset configuration scripts and model architecture used in this repository are borrowed from [here](https://github.com/IrohXu/lanenet-lane-detection-pytorch). 

## Implementation Background
The lane segmentation model used in this repository is trained on the open-source TuSimple dataset, obtained from [here](https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download). This repository contains the necessary code to work with the LaneNet instance segmentation model using the DeepLabV3+ encoder. 

In addition to training the model on the TuSimple dataset, we incorporate custom images (with a custom label generation implementation) of poor lane-marked rural roads and Speedway. The objective is to train LaneNet on a combination of the TuSimple and custom data and evaluate its performance on unseen samples across these categories.

## Dataset Configuration
1. Download the [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download) dataset and unzip it.
2. Once the TuSimple dataset has been extracted, add the custom dataset we created (i.e. rural) to ensure that the path to the datasets resembles the following structure: 
path/to/your/unzipped/file should like this:  
```
|--dataset
|----clips
|----label_data_0313.json
|----label_data_0531.json
|----label_data_0601.json
|----label_custom.json
```
3. Next, run the following commands to generate the training, validation, and test sets:

```
python tusimple_transform.py --src_dir path/to/your/unzipped/file --val True --test True
```

## Environment   
The environment for training and evaluation:  
```
python=3.6
torch>=1.2
numpy=1.7
torchvision>=0.4.0
matplotlib
opencv-python
pandas
```

If you have Conda installed on your machine, you can create a conda environment using the yaml file.
```
conda env create -f environment.yml
```

## Training 
Use the model_train.py script to train the model.

1. In lines 16-17, edit the paths to the training and validation .txt files
2. In line 20, adjust the number of epochs for training as desired
3. Execute the script 


## Testing     
Use the model_test.py script to test the model's output on an image

1. Edit line 12 to the path for the saved best-performing model weights from training
2. Set the path to test image on line 14
3. Execute the script. The results will be stored in the /Test_Outputs directory 
 

## Results
Example results are shown below. More results can be seen in the `/Test_Outputs` directory.

![alt text](https://github.com/jhpark98/lanenet-lane-detection-pytorch/blob/main/Test_Outputs/overlay_1700001654.0073147.jpg?raw=true)
![alt text](https://github.com/jhpark98/lanenet-lane-detection-pytorch/blob/main/Test_Outputs/overlay_1700001682.5807674.jpg?raw=true)


 

## Reference:  
DeepLabv3+ Encoder and DeepLabv3+ decoder refer from https://github.com/YudeWang/deeplabv3plus-pytorch

TuSimple dataset acquired from https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download
