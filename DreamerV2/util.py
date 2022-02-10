import time
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_preprocess_func(env_name):
    if "Breakout" in env_name:
        return _preprocess_breakout
    else:
        raise NotImplementedError(
           f"Frame processor not implemeted for {env_name}")


def _preprocess_breakout(frame, th=40.):

    image = Image.fromarray(frame)
    image = image.crop((0, 38, 160, 198)).convert("L")
    #image = image.point(lambda v: 255 if v > th else 0)

    image = image.resize((64, 64))
    #image = image.point(lambda v: 255 if v > th else 0)

    image_scaled = np.array(image) / 255.0
    image_out = image_scaled[np.newaxis, ..., np.newaxis]  #: (1, 64, 64, 1)

    return image_out.astype(np.float32)


def get_font(size="regular"):
    if size == "regular":
        fontsize = 18
    elif size == "small":
        fontsize = 12
    else:
        raise NotImplementedError(size)

    if os.name == 'posix':
        fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        return ImageFont.truetype(fontpath, fontsize)
    elif os.name == 'nt':
        return ImageFont.truetype("arial.ttf", fontsize)


def vizualize_vae(img_in, img_out, r, disc, r_total):
    """
        img_in: (64, 64, 1)
        img_out: (64, 64, 1)
    """
    assert img_in.shape == (64, 64)
    assert img_out.shape == (64, 64)

    img_out[img_out > 1.0] = 1.0

    img_in = Image.fromarray(img_in * 255).resize((192, 192))
    img_out = Image.fromarray(img_out * 255).resize((192, 192))

    pl, pr = 15, 15
    pt, pb = 60, 30

    canvas = Image.new('RGB', (pl+192+pr+192+pr+120+pr, pt+192+pb), color="black")
    fnt = get_font()

    canvas.paste(img_in, (pl, pt))
    canvas.paste(img_out, (pl+192+pr, pt))

    draw = ImageDraw.Draw(canvas)
    draw.text((pl, 30), "Original image", font=fnt, fill="white")
    draw.text((pl+192+pr, 30), "Reconstructed image", font=fnt, fill="white")

    r = round(r, 2)
    disc = round(disc, 2)
    r_total = round(r_total, 2)

    draw.text((pl+192+pr+192+pr, 50),
              f"R_pred: {r}", font=fnt, fill="white")
    draw.text((pl+192+pr+192+pr, 70),
              f"γ_pred: {disc}", font=fnt, fill="white")

    draw.text((pl+192+pr+192+pr, 110),
              f"Rtotal: {r_total}", font=fnt, fill="white")

    return canvas


def visualize_dream(img_outs, actions, rewards, discounts):

    #: actions of Breakout
    action_dict = {0: "NoOp", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}

    fnt = get_font()

    images = []

    pl, pr = 15, 15
    pt, pb = 30, 30

    for i, img in enumerate(img_outs):

        canvas = Image.new('RGB', (pl+192+pr+160+pr, pt+192+pb), color="black")

        img[img > 1.0] = 1.0
        frame = Image.fromarray(img * 255).resize((192, 192))
        canvas.paste(frame, (pl, pt))

        desc = Image.new('RGB', (160, pt+192+pb), color="black")
        draw = ImageDraw.Draw(desc)
        draw.text((0, 30), f"A: {action_dict[actions[i]]}", font=fnt, fill="white")
        draw.text((0, 50), f"R_pred: {round(rewards[i], 2)}", font=fnt, fill="white")
        draw.text((0, 70), f"γ_pred: {round(discounts[i], 2)}", font=fnt, fill="white")

        draw.text((0, 110), f"Time: {i}", font=fnt, fill="white")
        canvas.paste(desc, (pl+192+pr, pt))

        images.append(canvas)

    return images


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
    img = vizualize_vae(obs, obs)
