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

## Training the model    
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
 

 

## Reference:  
The lanenet project refers to the following research and projects:  
Neven, Davy, et al. "Towards end-to-end lane detection: an instance segmentation approach." 2018 IEEE intelligent vehicles symposium (IV). IEEE, 2018.   
```
@inproceedings{neven2018towards,
  title={Towards end-to-end lane detection: an instance segmentation approach},
  author={Neven, Davy and De Brabandere, Bert and Georgoulis, Stamatios and Proesmans, Marc and Van Gool, Luc},
  booktitle={2018 IEEE intelligent vehicles symposium (IV)},
  pages={286--291},
  year={2018},
  organization={IEEE}
}
```  
Paszke, Adam, et al. "Enet: A deep neural network architecture for real-time semantic segmentation." arXiv preprint arXiv:1606.02147 (2016).   
```
@article{paszke2016enet,
  title={Enet: A deep neural network architecture for real-time semantic segmentation},
  author={Paszke, Adam and Chaurasia, Abhishek and Kim, Sangpil and Culurciello, Eugenio},
  journal={arXiv preprint arXiv:1606.02147},
  year={2016}
}
```  
De Brabandere, Bert, Davy Neven, and Luc Van Gool. "Semantic instance segmentation with a discriminative loss function." arXiv preprint arXiv:1708.02551 (2017).   
```
@article{de2017semantic,
  title={Semantic instance segmentation with a discriminative loss function},
  author={De Brabandere, Bert and Neven, Davy and Van Gool, Luc},
  journal={arXiv preprint arXiv:1708.02551},
  year={2017}
}
```  
https://github.com/MaybeShewill-CV/lanenet-lane-detection    
https://github.com/klintan/pytorch-lanenet    

DeepLabv3+ Encoder and DeepLabv3+ decoder refer from https://github.com/YudeWang/deeplabv3plus-pytorch
