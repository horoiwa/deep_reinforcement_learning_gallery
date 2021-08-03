import time

import numpy as np
import tensorflow as tf
from PIL import Image


def get_preprocess_func(env_name):
    if "Breakout" in env_name:
        return _preprocess_breakout
    else:
        raise NotImplementedError(
           f"Frame processor not implemeted for {env_name}")


def _preprocess_breakout(frame):
    image = Image.fromarray(frame)
    image = image.convert("L").crop((0, 34, 160, 200)).resize((96, 96))
    image_scaled = np.array(image) / 255.0
    return image_scaled.astype(np.float32)


def value_rescaling(x):
    """https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/agents/r2d2/learner.py#L180
    """
    eps = 0.001
    return tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1.) - 1.) + eps * x


def inverse_value_rescaling(x):
    """https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/agents/r2d2/learner.py#L186
    """
    eps = 0.001
    return tf.math.sign(x) * (
        tf.math.square(
            ((tf.math.sqrt(1. + 4. * eps * (tf.math.abs(x) + 1. + eps))) - 1.) / (2. * eps)
            ) - 1.)


class Timer:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        fin = time.time() - self.start
        print(self.name, fin)


if __name__ == "__main__":
    import numpy as np
    arr = np.array([1,2,3,4],dtype=np.float32).reshape(4, 1)
    arr_inv = inverse_value_rescaling(arr)
    print(arr_inv)
