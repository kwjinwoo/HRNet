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


# augmentation
# random flip
def augmentation(x, y):
    random_seed = tf.random.uniform(shape=[1], minval=0., maxval=1.)

    if random_seed > 0.5:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
    return x, y


# get dataset function
def get_pascal_context_dataset(data_path, data_type, width, height, batch_size):
    reader_function = get_reader_function(width, height)
    ds = tf.data.TFRecordDataset(data_path).map(reader_function)

    if data_type == 'train':
        ds = ds.map(preprocessing).map(augmentation).batch(batch_size)
    else:
        ds = ds.map(preprocessing).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds
