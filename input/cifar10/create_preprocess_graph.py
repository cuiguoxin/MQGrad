
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf

label_bytes = 1;
image_bytes = 3072;
with tf.Session() as sess:
    record_bytes = tf.placeholder(tf.uint8, shape=(3073), name="raw_tensor")
    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32, name="label")
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [3, 32, 32])
    # Convert from [depth, height, width] to [height, width, depth].
    uint8image = tf.transpose(depth_major, [1, 2, 0])
    reshaped_image = tf.cast(uint8image, tf.float32)

    height = 28
    width = 28

    # Image processing for evaluation.
    # # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                            width, height)
    print resized_image.shape

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    print float_image.name
    print label.name
    print record_bytes.name

    tf.train.write_graph(sess.graph_def, './', 'preprocess.pb', as_text=False)

