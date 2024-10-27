import gym


def main():
    env = gym.make("Pendulum-v0")
    state = env.reset()
    while True:
        action = [1]
        next_state, reward, done, _ = env.step(state)

        print(state, next_state)

        if done:
            break
        else:
            state = next_state


if __name__ == "__main__":
    main()
