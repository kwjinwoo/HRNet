import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
from glob import glob
import random


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


class PASCALContextMaker:
    def __init__(self, img_dir, mat_path, origin_path, target_path):
        self.img_dir = img_dir
        self.mat_list = glob(mat_path)
        self.origin_dict = txt_to_dict(origin_path, 'origin')
        self.target_dict = txt_to_dict(target_path, 'target')
        self.train_size = 4998

        self.train_mat_list = self.mat_list[:self.train_size]
        self.val_mat_list = self.mat_list[self.train_size:]

    def load_image(self, path):
        file_name = os.path.basename(path).replace('.mat', '.jpg')
        image = tf.io.decode_jpeg(tf.io.read_file(os.path.join(self.img_dir, file_name)))
        return image

    def make_tfrecord(self, save_path, dataset_type):
        if dataset_type == 'train':
            mat_list = self.train_mat_list
        elif dataset_type == 'val':
            mat_list = self.val_mat_list
        else:
            raise 'Unexpected dataset_type(\'train\' or \'val\')'

        with tf.io.TFRecordWriter(save_path) as f:
            for mat_path in tqdm(mat_list):
                image = self.load_image(mat_path)
                image = tf.io.encode_jpeg(tf.cast(image, dtype=tf.uint8)).numpy()

                label = np.asarray(loadmat(mat_path)['LabelMap'])
                label = label_convert(self.target_dict, self.origin_dict, label)
                label = np.expand_dims(label, axis=-1)
                label = tf.io.encode_png(tf.cast(label, dtype=tf.uint8)).numpy()

                record = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                            'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
                        }
                    )
                )
                f.write(record.SerializeToString())


class PASCALContextLoader:
    def __init__(self, train_path, val_path, height, width, batch_size):
        self.train_path = train_path
        self.val_path = val_path
        self.height = height
        self.width = width
        self.batch_size = batch_size

    @staticmethod
    def get_reader_function():
        @tf.function
        def tfrecord_reader(example):
            feature_description = {"image": tf.io.VarLenFeature(dtype=tf.string),
                                   "mask": tf.io.VarLenFeature(dtype=tf.string)}

            example = tf.io.parse_single_example(example, feature_description)

            image = tf.sparse.to_dense(example['image'])[0]
            image = tf.io.decode_jpeg(image, channels=3)

            mask = tf.sparse.to_dense(example['mask'])[0]
            mask = tf.io.decode_png(mask, channels=1)
            return image, mask

        return tfrecord_reader

    @staticmethod
    def get_map_functions(height, width):
        def resize_and_normalize(x, y):
            x = tf.image.resize(x, (height, width)) / 255.
            y = tf.image.resize(y, (height, width),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return x, y
        def augmentation(x, y):
            random_flip = tf.random.uniform(shape=[1], minval=0., maxval=1.)
            random_crop = tf.random.uniform(shape=[1], minval=0., maxval=1.)

            if random_flip > 0.5:
                x = tf.image.flip_up_down(x)
                y = tf.image.flip_up_down(y)

            if random_crop > 0.5:
                image_shape = tf.shape(x)
                seed = random.randint(0, 42)
                x = tf.image.random_crop(x, (image_shape[0] // 2, image_shape[1] // 2, 3), seed=seed)
                y = tf.image.random_crop(y, (image_shape[0] // 2, image_shape[1] // 2, 1), seed=seed)
            return x, y
        return resize_and_normalize, augmentation

    def get_dataset(self, data_path, data_type):
        reader_function = self.get_reader_function()
        preprocess_func, augmentation_func = self.get_map_functions(self.height, self.width)

        ds = tf.data.TFRecordDataset(data_path).map(reader_function)

        if data_type == 'train':
            ds = ds.map(augmentation_func).map(preprocess_func).batch(self.batch_size)
        elif data_type == 'val':
            ds = ds.map(preprocess_func).batch(self.batch_size)
        else:
            raise 'Unexpected dataset_type(\'train\' or \'val\')'

        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def get_train_val_ds(self):
        train_ds = self.get_dataset(self.train_path, 'train')
        val_ds = self.get_dataset(self.val_path, 'val')

        return train_ds, val_ds