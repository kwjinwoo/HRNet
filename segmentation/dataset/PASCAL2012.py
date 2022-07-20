import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
from PIL import Image


class PASCAL2012Maker:
    def __init__(self, img_dir, label_dir, train_txt_path, val_txt_path):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.train_list = self.txt_to_list(train_txt_path)
        self.val_list = self.txt_to_list(val_txt_path)

    def load_label(self, file_name):
        label_path = os.path.join(self.label_dir, file_name + '.png')
        label = Image.open(label_path)
        label = np.array(label)
        label = np.where(label == 255, 21, label)
        return label

    def load_image(self, file_name):
        img_path = os.path.join(self.img_dir, file_name + '.jpg')
        img = tf.io.decode_jpeg(tf.io.read_file(img_path), channels=3)
        return img

    @staticmethod
    def txt_to_list(txt_path):
        txt_file = open(txt_path, 'r')
        return_list = []
        for line in txt_file.readlines():
            return_list.append(line.strip())
        return return_list

    def make_tfrecord(self, save_path, dataset_type):
        if dataset_type == 'train':
            name_list = self.train_list
        elif dataset_type == 'val':
            name_list = self.val_list
        else:
            raise 'Unexpected dataset_type(\'train\' or \'val\')'

        with tf.io.TFRecordWriter(save_path) as f:
            for name in tqdm(name_list):
                image = self.load_image(name)
                image = tf.io.encode_jpeg(tf.cast(image, dtype=tf.uint8)).numpy()

                label = self.load_label(name)
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


class PASCAL2012Loader:
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
            random_seed = tf.random.uniform(shape=[1], minval=0., maxval=1.)

            if random_seed > 0.5:
                x = tf.image.flip_up_down(x)
                y = tf.image.flip_up_down(y)
            return x, y

        return resize_and_normalize, augmentation

    def get_dataset(self, data_path, data_type):
        reader_function = self.get_reader_function()
        preprocess_func, augmentation_func = self.get_map_functions(self.height, self.width)

        ds = tf.data.TFRecordDataset(data_path).map(reader_function)

        if data_type == 'train':
            ds = ds.map(preprocess_func).map(augmentation_func).batch(self.batch_size)
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
