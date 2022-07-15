import tensorflow as tf
import numpy as np
import math


# tfrecorder reader
@tf.function
def tfrecord_reader(example):
    feature_description = {"image": tf.io.VarLenFeature(dtype=tf.string),
                           "points": tf.io.VarLenFeature(dtype=tf.int64)}

    example = tf.io.parse_single_example(example, feature_description)
    image_raw = tf.sparse.to_dense(example["image"])[0]
    image = tf.io.decode_image(image_raw, channels=3)

    points = tf.sparse.to_dense(example["points"])
    heatmaps = []
    for i in range(0, 51, 3):
        point = points[i:i+3]
        if point[-1] == 0:
            heatmap = tf.cast(tf.zeros(shape=(256 // 4, 192 // 4)), dtype=tf.float32)
        else:
            heatmap = tf.cast(gaussian(point[0], point[1], 256 // 4, 192 // 4), dtype=tf.float32)
        heatmaps.append(heatmap)
    heatmaps = tf.stack(heatmaps, -1)
    # heatmaps = tf.transpose(heatmaps, [1, 2, 0])
    return image, heatmaps


# make gaussian heatmap
def gaussian(xL, yL, H, W, sigma=2):

    channel = [tf.math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
    # channel = np.array(channel, dtype=np.float32)
    channel = tf.reshape(channel, (H, W))

    return channel


def get_pose_estimation_dataset(data_path):
    ds = tf.data.TFRecordDataset(data_path).map(tfrecord_reader)
    return ds