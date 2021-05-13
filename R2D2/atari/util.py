import numpy as np
import tensorflow as tf
from PIL import Image


def value_function_rescaling(x):
    """https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/agents/r2d2/learner.py#L180
    """
    eps = 0.001
    return tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1.) - 1.) + eps * x


def inverse_value_function_rescaling(x):
    """https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/agents/r2d2/learner.py#L186
    """
    eps = 0.001
    return tf.math.sign(x) * (
        tf.math.square(
            ((tf.math.sqrt(1. + 4. * eps * (tf.math.abs(x) + 1. + eps))) - 1.) / (2. * eps)
            ) - 1.)


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


def get_initial_lives(env_name):
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


