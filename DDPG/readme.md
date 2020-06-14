
[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)


DQNは高次元の観測空間の問題を解決しますが、DQNは離散および低次元のアクション空間しか処理できません。

DQNは、アクション値関数を最大化するアクションを見つけることに依存しているため、連続ドメインに直接適用することはできません。

DDPGは off-policy Actor-Critic


A2CもDDPGも方策勾配だが、A2CはOn-PolicyでDDPGはOff-policy

この差は方策更新の方向決め関数の差である

Adavantage関数(A2C)はポリシーに基づいた行動した実際の結果が必要なので再計算できず、使い捨て

Q関数(DDPG)はあとから再計算できる


