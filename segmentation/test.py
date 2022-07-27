import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import argparse


parser = argparse.ArgumentParser(description='inference segmentation')
parser.add_argument('--model_path', type=str, default="./model_asset")
parser.add_argument('--save_dir', type=str, default="./infer_img")
parser.add_argument('--input_dir', type=str, default="./input_img")


args = parser.parse_args()


def load_img(path, h, w):
    img = tf.io.decode_image(tf.io.read_file(path))
    input_img = tf.image.resize(img, (h, w)) / 255.
    input_img = tf.expand_dims(input_img, axis=0)

    return input_img, img.numpy()


def predict(input_img, model):
    pred = model.predict(input_img)[0]
    pred = np.argmax(pred, axis=-1)
    return pred


def visualization(pred, origin, save_path):
    plt.figure(figsize=(16, 16))
    plt.subplot(1, 2, 1)
    plt.imshow(origin)
    plt.axis('off')
    plt.title('input image', size=15)

    plt.subplot(1, 2, 2)
    plt.imshow(pred)
    plt.axis('off')
    plt.title('segmentation', size=15)

    plt.savefig(save_path)


if __name__ == '__main__':
    model_path = args.model_path
    input_dir = args.input_dir
    save_dir = args.save_dir

    model = load_model(model_path)
    _, height, width, channel = model.input.shape
    img_list = glob(os.path.join(input_dir, '*'))

    for img_path in tqdm(img_list):
        file_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, file_name)

        input_image, origin_image = load_img(img_path, height, width)
        pred = predict(input_image, model)
        visualization(pred, origin_image, save_path)




