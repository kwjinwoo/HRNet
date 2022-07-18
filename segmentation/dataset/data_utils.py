import tensorflow as tf


# tfrecoder reader getter function
def get_reader_function(width, height):
    @tf.function
    def tfrecord_reader(example):
        feature_description = {"image": tf.io.VarLenFeature(dtype=tf.string),
                               "mask": tf.io.VarLenFeature(dtype=tf.int64)}

        example = tf.io.parse_single_example(example, feature_description)
        image_raw = tf.sparse.to_dense(example["image"])[0]
        image = tf.io.decode_image(image_raw, channels=3)

        label = tf.sparse.to_dense(example["mask"])
        label = tf.reshape(label, (height, width, 1))
        return image, label
    return tfrecord_reader


# preprocessing
# image normalizing(0-1)
def preprocessing(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    return x, y


# get dataset function
def get_pascal_context_dataset(data_path, width, height, batch_size):
    reader_function = get_reader_function(width, height)
    ds = tf.data.TFRecordDataset(data_path).map(reader_function)
    ds = ds.map(preprocessing).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds
