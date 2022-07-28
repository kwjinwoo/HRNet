# HRNet Semantic Segmentation
* HRNet Semantic Segmentation task Test
* Using [OXFORD](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset

## train setting
```
input_image_widht = 480
input_imgae_height = 480
input_image_channel = 3
HRNet_high_resolution_featuremap_channel(c) = 48
batch_size = 16
optimizer = Adam
learning_rate = 0.001
weight_decay = 0.0001
```
## result
* train image   
![TRAIN1](/segmentation/infer_img/train1.jpg)   
![TRAIN2](/segmentation/infer_img/train2.jpg)
* valid image   
![VAL1](/segmentation/infer_img/val1.jpg)   
![VAL2](/segmentation/infer_img/val2.jpg)
* test image   
![TEST](/segmentation/infer_img/alexandru-rotariu-o_QTeyGVWjQ-unsplash.jpg)