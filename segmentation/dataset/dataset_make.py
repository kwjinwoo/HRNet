import argparse
import os
from PASCAL2012 import PASCAL2012Maker
from OXFORD import OXFORDMaker
from PASCALContext import PASCALContextMaker


parser = argparse.ArgumentParser(description='Make dataset TFRecord file')
parser.add_argument('--dataset', type=str, required=True,
                    help='dataset name')
parser.add_argument('--img_dir', type=str, required=True,
                    help='jpg images directory path')
parser.add_argument('--label_dir', type=str, required=True,
                    help='png labels directory path')
parser.add_argument('--train_txt', type=str, required=True,
                    help='train.txt file path')
parser.add_argument('--val_txt', type=str, required=True,
                    help='val.txt file path')
parser.add_argument('--shuffle', type=bool, default=False)
parser.add_argument('--save_dir', type=str, default='./tfrecords/', help='the directory path to saved tfrecord')

args = parser.parse_args()


if __name__ == '__main__':
    dataset_type = args.dataset
    print()
    if dataset_type == 'pascal':
        maker = PASCAL2012Maker(args.img_dir, args.label_dir, args.train_txt, args.val_txt)

        # train dataset
        save_path = os.path.join(args.save_dir, 'train_PASCAL2012.tfrecord')
        maker.make_tfrecord(save_path, 'train')

        # val dataset
        save_path = os.path.join(args.save_dir, 'val_PASCAL2012.tfrecord')
        maker.make_tfrecord(save_path, 'val')
    elif dataset_type == 'oxford':
        maker = OXFORDMaker(args.img_dir, args.label_dir, args.train_txt, args.val_txt, args.shuffle)

        # train dataset
        save_path = os.path.join(args.save_dir, 'train_OXFORD.tfrecord')
        maker.make_tfrecord(save_path, 'train')

        # val dataset
        save_path = os.path.join(args.save_dir, 'val_OXFORD.tfrecord')
        maker.make_tfrecord(save_path, 'val')

    elif dataset_type == 'context':
        maker = PASCALContextMaker(args.img_dir, args.label_dir, args.train_txt, args.val_txt)

        # train dataset
        save_path = os.path.join(args.save_dir, 'train_PASCALContext.tfrecord')
        maker.make_tfrecord(save_path, 'train')

        # val dataset
        save_path = os.path.join(args.save_dir, 'val_PASCALContext.tfrecord')
        maker.make_tfrecord(save_path, 'val')
