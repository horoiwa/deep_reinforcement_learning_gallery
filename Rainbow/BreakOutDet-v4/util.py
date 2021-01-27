import tensorflow as tf


def preprocess_frame(frame):

    image = tf.cast(tf.convert_to_tensor(frame), tf.float32)
    image_gray = tf.image.rgb_to_grayscale(image)
    image_crop = tf.image.crop_to_bounding_box(image_gray, 34, 0, 160, 160)
    image_resize = tf.image.resize(image_crop, [84, 84])
    image_scaled = tf.divide(image_resize, 255)

    frame = image_scaled.numpy()[:, :, 0]

    return frame
