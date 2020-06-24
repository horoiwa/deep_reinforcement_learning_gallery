
[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)


DQNは高次元の観測空間の問題を解決しますが、DQNは離散および低次元のアクション空間しか処理できません。

DQNは、アクション値関数を最大化するアクションを見つけることに依存しているため、連続ドメインに直接適用することはできません。

DDPGは off-policy Actor-Critic


A2CもDDPGも方策勾配だが、A2CはOn-PolicyでDDPGはOff-policy

この差は方策更新の方向決め関数の差である

Adavantage関数(A2C)はポリシーに基づいた行動した実際の結果が必要なので再計算できず、使い捨て

Q関数(DDPG)はあとから再計算できる



Actorは0.0001, Criticは0.001

L2 weight decay(kernel regularizer 0.01)

soft targetは
target_net.set_weight(0.005 * main.get_weight + (1-0.005) * target.get_weight)


actorはrelu活性化だがさいごだけtanh



最近の結果では、相関のない、平均ゼロのガウスノイズが完全に機能することが示されています。後者の方が簡単なので、推奨されます。より高品質のトレーニングデータを取得しやすくするために、トレーニング中にノイズのスケールを小さくすることができます。（この実装ではこれを行わず、ノイズスケールを全体にわたって固定します。）

探索ノイズはN(0, 0.1)で固定でOK

https://towardsdatascience.com/teach-your-ai-how-to-walk-5ad55fce8bca


noise

https://openai.com/blog/better-exploration-with-parameter-noise/
