import tensorflow as tf


def frame_preprocess(frame):

    def _frame_preprocess(frame):
        """Breakout向けの切り取りであることに注意
        """
        image = tf.cast(tf.convert_to_tensor(frame), tf.float32)
        image_gray = tf.image.rgb_to_grayscale(image)
        image_crop = tf.image.crop_to_bounding_box(image_gray, 34, 0, 160, 160)
        image_resize = tf.image.resize(image_crop, [84, 84])
        image_scaled = tf.divide(image_resize, 255)
        return image_scaled

    frame = _frame_preprocess(frame).numpy()[:, :, 0]

    return frame
