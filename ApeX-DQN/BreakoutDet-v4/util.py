import time

import numpy as np
from PIL import Image


class Timer:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        fin = time.time() - self.start
        print(self.name, fin)

def preprocess_frame(frame):
    """Breakout only"""
    image = Image.fromarray(frame)
    image = image.convert("L").crop((0, 34, 160, 200)).resize((84, 84))
    image_scaled = np.array(image) / 255.0
    return image_scaled.astype(np.float32)
