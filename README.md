# HRNet Implementation
## Introduction
* This repository is HRNet Implementation code with Tensorflow
* Implemented Simple Semantic Segmentation task
## requirement
    python == 3.8
    tensorflow == 2.8.0
    matplotlib == 3.5.2
## train
* Semantic Segmentation
    ```
    python segmentation_train.py --dataset_type --width --height --num_class --c --batch_size --num_epoch --initial_lr --weight_decay   

    args
    dataset_type : your dataset type(oxford, context, pascal)
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