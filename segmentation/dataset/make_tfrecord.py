import tensorflow as tf
from scipy.io import loadmat
from glob import glob
import numpy as np
import os
from tqdm import tqdm
import random
import argparse


parser = argparse.ArgumentParser(description='Make TFRecord file')
parser.add_argument('--width', type=int, required=True, help='input and output image width')
parser.add_argument('--height', type=int, required=True, help='input and output image height')

args = parser.parse_args()


# label txt file convert to dictionary
def txt_to_dict(path, txt_type):
    labels = open(path, 'r')

    label_dict = {}
    for line in labels.readlines():
        if ':' not in line:
            continue
        id_, label = line.strip().split(':')

        # about target(59 labels), key is label name
        if txt_type == 'target':
            label_dict[label.strip()] = int(id_)

        # about origin(400+ labels), key is id
        else:
            label_dict[int(id_)] = label.strip()

    return label_dict


# origin label and id convert to target label and id
def label_convert(target, origin, mat):
    convert_mask = mat.copy()

    ids = np.unique(mat)
    for id_ in ids:
        indices = np.where(convert_mask == id_)
        label = origin[id_]

        if label in target.keys():
            convert_id = target[label]
        # background
        else:
            convert_id = 0
        convert_mask[indices] = convert_id

    return convert_mask


if __name__ == '__main__':
    img_dir = './VOCdevkit/VOC2010/JPEGImages'
    mat_list = glob('./trainval/trainval/*')
    random.shuffle(mat_list)
    print('total mat files :', len(mat_list))

    train_size = int(len(mat_list) * 0.85)
    train_mat_list = mat_list[:train_size]
    val_mat_list = mat_list[train_size:]

    target_path = './59_labels.txt'
    origin_path = './trainval/labels.txt'

    target_dict = txt_to_dict(target_path, 'target')
    origin_dict = txt_to_dict(origin_path, 'origin')

    target_width = args.width
    target_height = args.height

    save_path = './train_PASCAL_context_{}_{}.tfrecord'.format(target_width, target_height)
    with tf.io.TFRecordWriter(save_path) as f:
        for mat_path in tqdm(train_mat_list):
            img_file_name = os.path.basename(mat_path).replace('.mat', '.jpg')

            mat_file = loadmat(mat_path)
            mat_label = np.asarray(mat_file['LabelMap'])

            converted_mat_label = label_convert(target_dict, origin_dict, mat_label)
            converted_mat_label = np.expand_dims(converted_mat_label, axis=-1)
            converted_mat_label = tf.image.resize(converted_mat_label, (target_height, target_width),
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy()
            converted_mat_label = converted_mat_label.reshape(-1, )    # flatten

            img = tf.io.decode_jpeg(tf.io.read_file(os.path.join(img_dir, img_file_name)))
            img = tf.image.resize(img, (target_height, target_width))
            img = tf.io.encode_jpeg(tf.cast(img, dtype=tf.uint8)).numpy()

            record = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                        'mask': tf.train.Feature(int64_list=tf.train.Int64List(value=converted_mat_label))
                    }
                )
            )
            f.write(record.SerializeToString())

    save_path = './val_PASCAL_context_{}_{}.tfrecord'.format(target_width, target_height)
    with tf.io.TFRecordWriter(save_path) as f:
        for mat_path in tqdm(val_mat_list):
            img_file_name = os.path.basename(mat_path).replace('.mat', '.jpg')

            mat_file = loadmat(mat_path)
            mat_label = np.asarray(mat_file['LabelMap'])

            converted_mat_label = label_convert(target_dict, origin_dict, mat_label)
            converted_mat_label = np.expand_dims(converted_mat_label, axis=-1)
            converted_mat_label = tf.image.resize(converted_mat_label, (target_height, target_width),
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy()
            converted_mat_label = converted_mat_label.reshape(-1, )    # flatten

            img = tf.io.decode_jpeg(tf.io.read_file(os.path.join(img_dir, img_file_name)))
            img = tf.image.resize(img, (target_height, target_width))
            img = tf.io.encode_jpeg(tf.cast(img, dtype=tf.uint8)).numpy()

            record = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                        'mask': tf.train.Feature(int64_list=tf.train.Int64List(value=converted_mat_label))
                    }
                )
            )
            f.write(record.SerializeToString())



