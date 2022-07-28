from segmentation.dataset.PASCAL2012 import PASCAL2012Loader
from segmentation.dataset.OXFORD import OXFORDLoader
from segmentation.dataset.PASCALContext import PASCALContextLoader
import models
import tensorflow as tf
import os
import argparse


parser = argparse.ArgumentParser(description='Training Segmentation model')
parser.add_argument('--dataset_type', type=str, required=True)
parser.add_argument('--width', type=int, required=True, help='input and output image width')
parser.add_argument('--height', type=int, required=True, help='input and output image height')
parser.add_argument('--num_class', type=int, required=True, help='output class number')
parser.add_argument('--c', type=int, required=True, help='the number of channel of high resolution feature map')
parser.add_argument('--batch_size', type=int, required=True, help='dataset batch size')
parser.add_argument('--num_epoch', type=int, required=True, help='the number of epoch')
parser.add_argument('--initial_lr', type=float, required=True, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, required=True, help='weight decay value')

args = parser.parse_args()


if __name__ == '__main__':
    width = args.width
    height = args.height
    num_class = args.num_class
    c = args.c
    model = models.HRNet((height, width, 3), c).build_hrnet('segmentation', num_class, args.weight_decay)
    print(model.summary())

    if args.dataset_type == 'context':
        batch_size = args.batch_size
        loader = PASCALContextLoader(train_path='./segmentation/dataset/tfrecords/train_PASCALContext.tfrecord',
                                     val_path='./segmentation/dataset/tfrecords/val_PASCALContext.tfrecord',
                                     height=height, width=width, batch_size=batch_size)
        train_ds, val_ds = loader.get_train_val_ds()
    elif args.dataset_type == 'pascal':
        batch_size = args.batch_size
        loader = PASCAL2012Loader(train_path='./segmentation/dataset/tfrecords/train_PASCAL2012.tfrecord',
                                     val_path='./segmentation/dataset/tfrecords/val_PASCAL2012.tfrecord',
                                     height=height, width=width, batch_size=batch_size)
        train_ds, val_ds = loader.get_train_val_ds()

    elif args.dataset_type == 'oxford':
        batch_size = args.batch_size
        loader = OXFORDLoader(train_path='./segmentation/dataset/tfrecords/train_OXFORD.tfrecord',
                                  val_path='./segmentation/dataset/tfrecords/val_OXFORD.tfrecord',
                                  height=height, width=width, batch_size=batch_size)
        train_ds, val_ds = loader.get_train_val_ds()

    # learning rate scheduling
    decay_steps = 10000
    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=args.initial_lr,
        decay_steps=decay_steps,
        end_learning_rate=args.initial_lr * 0.01,
        power=0.9,
        cycle=True
    )

    # optimizer and loss
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=0.9)
    # optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # callback function
    callbacks_list = [tf.keras.callbacks.ModelCheckpoint(
        filepath='./segmentation/ckpt'+args.dataset_type+'/'+args.dataset_type+'_segmentation',
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
        save_best_only=True,
        verbose=1)]

    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.num_epoch, callbacks=callbacks_list,
                     verbose=2)

    model_save_path = './segmentation/model_asset/'+args.dataset_type
    if os.path.isdir(model_save_path):
        model.save(model_save_path)
    else:
        os.mkdirs(model_save_path)
        model.save(model_save_path)


