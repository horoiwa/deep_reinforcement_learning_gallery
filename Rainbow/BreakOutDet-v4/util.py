import tensorflow as tf


def preprocess_frame(frame):

    image = tf.cast(tf.convert_to_tensor(frame), tf.float32)
    image_gray = tf.image.rgb_to_grayscale(image)
    image_crop = tf.image.crop_to_bounding_box(image_gray, 34, 0, 160, 160)
    image_resize = tf.image.resize(image_crop, [84, 84])
    image_scaled = tf.divide(image_resize, 255)

    frame = image_scaled.numpy()[:, :, 0]

    return frame


@tf.function
def huber_loss(td_errors, d=1.0):
    """
    See https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/losses.py#L1098-L1162
    """
    is_smaller_than_d = tf.abs(td_errors) < d
    squared_loss = 0.5 * tf.square(td_errors)
    linear_loss = 0.5 * d ** 2 + d * (tf.abs(td_errors) - d)
    loss = tf.where(is_smaller_than_d, squared_loss, linear_loss)
    loss = tf.reduce_mean(loss)
    return loss
