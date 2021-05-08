import numpy as np
from PIL import Image


def get_preprocess_func(env_name):
    if "Breakout" in env_name:
        return _preprocess_breakout
    elif "Pacman" in env_name:
        return _preprocess_mspackman
    else:
        raise NotImplementedError(
           f"Frame processor not implemeted for {env_name}")


def _preprocess_breakout(frame):
    image = Image.fromarray(frame)
    image = image.convert("L").crop((0, 34, 160, 200)).resize((84, 84))
    image_scaled = np.array(image) / 255.0
    return image_scaled.astype(np.float32)


def _preprocess_mspackman(frame):
    image = Image.fromarray(frame)
    image = image.convert("L").crop((0, 0, 160, 170)).resize((84, 84))
    image_scaled = np.array(image) / 255.0
    return image_scaled.astype(np.float32)


def get_lives(env_name):
    if "Breakout" in env_name:
        return 5
    elif "Pacman" in env_name:
        return 3
    else:
        raise NotImplementedError(
           f"Frame processor not implemeted for {env_name}")


if __name__ == "__main__":
    import gym

    #env_name = "BreakoutDeterministic-v4"
    env_name = "MsPacmanDeterministic-v4"
    env = gym.make(env_name)

    preprcess_func = get_preprocess_func(env_name)
    frame = env.reset()
    frame_processed = preprcess_func(frame)

    Image.fromarray(frame).show()
    print(frame.shape)
    Image.fromarray(frame_processed * 255).show()


