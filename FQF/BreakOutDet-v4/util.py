import numpy as np
from PIL import Image


def frame_preprocess(frame):
    """Breakout only"""
    image = Image.fromarray(frame)
    image = image.convert("L").crop((0, 34, 160, 200)).resize((84, 84))
    image = np.array(image) / 255.0
    return image.astype(np.float32)
