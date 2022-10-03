import numpy as np
from PIL import Image


def preprocess(frames):
    """
    Transforms two frames into a Nature DQN observation by pool and resize
    See https://github.com/google/dopamine/dopamine/discrete_domains/atari_lib.py
    """

    arr = np.maximum(frames[-1], frames[-2])
    image = Image.fromarray(arr)
    image = image.convert("L").resize((84, 84))

    return np.array(image, dtype=np.float32)
