import time
import numpy as np
from PIL import Image


def get_preprocess_func(env_name):
    if "Breakout" in env_name:
        return _preprocess_breakout
    else:
        raise NotImplementedError(
           f"Frame processor not implemeted for {env_name}")


def _preprocess_breakout(frame):
    image = Image.fromarray(frame)
    image = image.convert("L").crop((0, 34, 160, 200)).resize((64, 64))
    image_scaled = np.array(image) / 255.0
    image_out = image_scaled[np.newaxis, ..., np.newaxis]  #: (1, 64, 64, 1)
    return image_out.astype(np.float32)


class Timer:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        fin = time.time() - self.start
        print(self.name, fin)
