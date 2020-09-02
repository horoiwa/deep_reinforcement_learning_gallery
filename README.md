# DeepRL_TF2

Deep reinforcement learning sample codes with tensorflow2

深層強化学習の主要な手法をtensorflow2.Xで実装します。

どの実装も1フォルダで完結するようにしているが、実装した実装した時期が古いものほど酷いコードになっていることには注意。


## Requirements

`python==3.7` and `tensorflow==2.1.0`

<br>

## 参考文献

### DQN (Deep Q Network)

[Playing Atari with Deep Reinforcement Learning (2013)](https://arxiv.org/abs/1312.5602)

[Human-level control through deep reinforcement learning (2015)](https://www.nature.com/articles/nature14236.)


[Implementing the Deep Q-Network](https://arxiv.org/pdf/1711.07478.pdf)


[How to match DeepMind’s Deep Q-Learning score in Breakout](https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756)


[Why could my ddqn get significantly worse after beating the game repeatedly](https://datascience.stackexchange.com/questions/56053/why-could-my-ddqn-get-significantly-worse-after-beating-the-game-repeatedly)

[OpenAI baselines DQN](https://openai.com/blog/openai-baselines-dqn/)

<br>

## Double DQN

[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

<br>

## Dueling Network

[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

<br>

## Prioritized Experience Replay

[original](https://arxiv.org/abs/1511.05952)

[baselines](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py)

## Ape-X

[Distributed Prioritized Experience Replay](https://arxiv.org/pdf/1803.00933.pdf)

### A3C/A2C (Advantage Actor Critic)

[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

[Deep Reinforcement Learning: Playing CartPole through Asynchronous Advantage Actor Critic (A3C) with tf.keras and eager execution](https://blog.tensorflow.org/2018/07/deep-reinforcement-learning-keras-eager-execution.html)


[OpenAI baselines](https://openai.com/blog/baselines-acktr-a2c/)


## DDPG

[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)

[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

## TD3

[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)

## TRPO

## PPO

### その他

[Reinforcement Learning for Improving Agent Design](https://arxiv.org/abs/1810.03779)


バッチ強化学習
[Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/abs/1812.02900)


### Atari emulator

https://www.retrogames.cz/play_222-Atari2600.php
