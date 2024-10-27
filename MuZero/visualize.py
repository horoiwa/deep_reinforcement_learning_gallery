import collections
from PIL import Image, ImageDraw, ImageFont

import gym

import util
from buffer import PrioritizedReplay
from actor import Actor
from networks import DynamicsNetwork, PVNetwork, RepresentationNetwork
from mcts import AtariMCTS


def visualize(env_id="BreakoutDeterministic-v4",
              n_frames=4, V_min=-30, V_max=30):

    env = gym.make(env_id)

    action_space = env.action_space.n

    preprocess_func = util.get_preprocess_func(env_id)

    frame = preprocess_func(env.reset())

    frame_history = collections.deque(
        [frame] * n_frames, maxlen=n_frames)

    action_history = collections.deque(
        [0] * n_frames, maxlen=n_frames)

    repr_network = RepresentationNetwork(
        action_space=action_space)
    repr_network.load_weights("checkpoints/repr_net")

    pv_network = PVNetwork(action_space=action_space,
                           V_min=V_min, V_max=V_max)
    pv_network.load_weights("checkpoints/pv_net")

    dynamics_network = DynamicsNetwork(action_space=action_space,
                                       V_min=V_min, V_max=V_max)
    dynamics_network.load_weights("checkpoints/dynamics_net")

    mcts = AtariMCTS(
        action_space=action_space,
        pv_network=pv_network,
        dynamics_network=dynamics_network,
        gamma=0.997,
        dirichlet_alpha=None)

    episode_rewards, episode_steps = 0, 0

    done = False

    images = []

    while not done:

        hidden_state, obs = repr_network.predict(frame_history, action_history)

        mcts_policy, action, root_value = mcts.search(
            hidden_state, 20, T=0.1)

        next_hidden_state, reward_pred = dynamics_network.predict(hidden_state, action)
        reward_pred = reward_pred.numpy()[0][0]

        frame, reward, done, info = env.step(action)

        print()
        print("STEP:", episode_steps)
        print("Reward", reward)
        print("Action", action)

        #: shape = (160, 210, 3)
        img_frame = Image.fromarray(frame)

        img_desc = Image.new('RGB', (280, 210), color="black")
        fnt = ImageFont.truetype("arial.ttf", 18)
        fnt_sm = ImageFont.truetype("arial.ttf", 12)

        pl = 30
        pb = 30

        v = str(round(root_value, 2))
        p = str([round(prob, 2) for prob in mcts_policy])
        r = str(round(reward_pred, 2))

        draw = ImageDraw.Draw(img_desc)
        draw.text((pl, 20), f"V(s): {v}", font=fnt, fill="white")
        draw.text((pl, 20+pb), f"R(s, a): {r}", font=fnt, fill="white")
        draw.text((pl, 20+pb*2), f"π(s): {p}", font=fnt, fill="white")

        draw.text((pl, 20+pb*3.5), f"Note:", font=fnt_sm, fill="white")
        draw.text((pl, 20+pb*4), "{ 0: Noop, 1: FIRE, 2: Left, 3: Right }",
                  font=fnt_sm, fill="white")

        img_bg = Image.new(
            'RGB', (img_frame.width + img_desc.width, img_frame.height))

        img_bg.paste(img_frame, (0, 0))
        img_bg.paste(img_desc, (img_frame.width, 0))

        images.append(img_bg)

        episode_rewards += reward

        episode_steps += 1

        frame_history.append(preprocess_func(frame))

        action_history.append(action)

    print()
    print("===="*5)
    print("FINISH")
    print(episode_steps, episode_rewards)

    images[0].save(
        'tmp/muzero.gif',
        save_all=True, append_images=images[1:],
        optimize=False, duration=60, loop=0)

    return episode_rewards, episode_steps


if __name__ == "__main__":
    visualize()
