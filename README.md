# HRNet Implementation
## Introduction
* This repository is HRNet Implementation code with Tensorflow
* Implemented Simple Semantic Segmentation task
## requirement
    python == 3.8
    tensorflow == 2.8.0
    matplotlib == 3.5.2
## dataset
* Semantic Segmentation   
using OXFORD PET Dataset
```
python ./segmentation/dataset/dataset_make.py --img_dir --label_dir --train_txt --val_txt --shuffle --save_dir
  
args
--img_dr : jpg images directory path
--label_dir : png labels directory path
--train_txt : train.txt file path
--val_txt : val.txt file path
--shuffle : data shuffle
--save_dir : the directory path saved tfrecord files
```
then tfrecord files generated
  ```
  $segmentation/
  ├── tfrecords
  |   ├── dataset_type.tfrecord
  |   └── dataset_type.tfrecord
  ```
## train
* Semantic Segmentation
```
python segmentation_train.py --width --height --num_class --c --batch_size --num_epoch --initial_lr --weight_decay   

args
width : input image width
height : input image height
num_class : the number of class
c : the high resolution feature map channels
batch_size : dataset batch size
num_epoch : the number of epoch
initial_lr : initial learning rate
weight_decay : weight decay ratio
 ```
## result
* [segmentation](https://github.com/kwjinwoo/HRNet/tree/main/segmentation)
## reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, 
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI.