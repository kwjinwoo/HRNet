import argparse
import os
from PASCAL_2012 import PASCAL2012


parser = argparse.ArgumentParser(description='Make PASCAL dataset TFRecord file')
parser.add_argument('--img_dir', type=str, default='./VOCdevkit/VOC2012/JPEGImages',
                    help='jpg images directory path')
parser.add_argument('--label_dir', type=str, default='./VOCdevkit/VOC2012/SegmentationClass',
                    help='png labels directory path')
parser.add_argument('--train_txt', type=str, default='./VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt',
                    help='train.txt file path')
parser.add_argument('--val_txt', type=str, default='./VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
                    help='val.txt file path')
parser.add_argument('--save_dir', type=str, default='./', help='the directory path to saved tfrecord')

args = parser.parse_args()


if __name__ == '__main__':
    pascal = PASCAL2012(args.img_dir, args.label_dir, args.train_txt, args.val_txt)

    # train dataset
    save_path = os.path.join(args.save_dir, 'train_PASCAL2012.tfrecord')
    pascal.make_tfrecord(save_path, 'train')

    # val dataset
    save_path = os.path.join(args.save_dir, 'val_PASCAL2012.tfrecord')
    pascal.make_tfrecord(save_path, 'val')
