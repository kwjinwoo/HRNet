import segmentation.dataset.data_utils as du
import models
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import argparse


parser = argparse.ArgumentParser(description='Training Segmentation model')
parser.add_argument('--width', type=int, required=True, help='input and output image width')
parser.add_argument('--height', type=int, required=True, help='input and output image height')
parser.add_argument('--num_class', type=int, required=True, help='output class number')
parser.add_argument('--c', type=int, required=True, help='the number of channel of high resolution feature map')
parser.add_argument('--batch_size', type=int, required=True, help='dataset batch size')
parser.add_argument('--num_epoch', type=int, required=True, help='the number of epoch')

args = parser.parse_args()


if __name__ == '__main__':
    width = args.width
    height = args.height
    num_class = args.num_class
    c = args.c
    model = models.HRNet((height, width, 3), c).build_hrnet('segmentation', num_class)
    print(model.summary())

    batch_size = args.batch_size
    train_ds = du.get_pascal_context_dataset('./segmentation/dataset/train_PASCAL_context_480_480.tfrecord',
                                             width, height, batch_size)
    val_ds = du.get_pascal_context_dataset('./segmentation/dataset/val_PASCAL_context_480_480.tfrecord',
                                           width, height, batch_size)

    # optimizer and loss
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # callback function
    callbacks_list = [tf.keras.callbacks.ModelCheckpoint(
        filepath='./segmentation/ckpt/PASCAL_segmentation',
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
        save_best_only=True,
        verbose=1)]

    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.num_epoch, callbacks=callbacks_list,
                     verbose=1)

    # plotting loss
    plt.figure(figsize=(12, 12))
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.legend()

    loss_img_dir = './segmentation/loss'
    loss_img_name = 'PASCAL_segmentation.png'

    if os.path.isdir(loss_img_dir):
        plt.savefig(os.path.join(loss_img_dir, loss_img_name))
    else:
        os.mkdir(loss_img_dir)
        plt.savefig(os.path.join(loss_img_dir, loss_img_name))
