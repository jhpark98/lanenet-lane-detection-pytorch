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

Generate training/ val/ test/ set:  
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

Use the following scripts to train the model.

Using example folder with ENet:   
```
python model_train.py --dataset ./data/training_data_example
```
Using tusimple folder with ENet/Focal loss:   
```
python model_train.py --dataset path/to/tusimpledataset/training
```
Using tusimple folder with ENet/Cross Entropy loss:   
```
python model_train.py --dataset path/to/tusimpledataset/training --loss_type CrossEntropyLoss
```
Using tusimple folder with DeepLabv3+:   
```
python model_train.py --dataset path/to/tusimpledataset/training --model_type DeepLabv3+
```    

If you want to change focal loss to cross entropy loss, do not forget to adjust the hyper-parameter of instance loss and binary loss in ./model/lanenet/train_lanenet.py    

## Testing result    
A pretrained trained model by myself is located in ./log (only trained in 25 epochs)      
Test the model:    
```
python model_test.py --img ./data/tusimple_test_image/0.jpg
```
 

 

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
