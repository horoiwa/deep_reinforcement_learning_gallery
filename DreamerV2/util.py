import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def vizualize_vae(img_in, img_out):
    """
        img_in: (64, 64, 1)
        img_out: (64, 64, 1)
    """
    assert img_in.shape == (1, 64, 64, 1)
    assert img_out.shape == (1, 64, 64, 1)

    img_in = Image.fromarray(img_in[0, :, :, 0] * 255).resize((192, 192))
    img_out = Image.fromarray(img_out[0, :, :, 0] * 255).resize((192, 192))

    pl, pr = 15, 15
    pt, pb = 60, 30

    canvas = Image.new('RGB', (pl+192+pr+192+pr, pt+192+pb), color="black")
    fnt = ImageFont.truetype("arial.ttf", 18)
    fnt_sm = ImageFont.truetype("arial.ttf", 12)

    canvas.paste(img_in, (pl, pt))
    canvas.paste(img_out, (pl+192+pr, pt))

    draw = ImageDraw.Draw(canvas)
    draw.text((pl, 30), f"Original image", font=fnt, fill="white")
    draw.text((pl+192+pr, 30), f"Reconstructed image", font=fnt, fill="white")

    return canvas


def visualize_dream():
    return None


class Timer:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        fin = time.time() - self.start
        print(self.name, fin)

if __name__ == '__main__':
    import gym
    env_id = "BreakoutDeterministic-v4"
    env = gym.make(env_id)
    preprocessor = get_preprocess_func(env_id)
    obs = preprocessor(env.reset())
    img = vizualize(obs, obs)
