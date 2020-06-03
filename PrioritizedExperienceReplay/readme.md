cartpoleはDQN＋PERだが、spaceinvadorsはDuelingDoubleDQN + PER


## 優先度つき経験再生

https://arxiv.org/abs/1511.05952


The central component of prioritized replay is the criterion by which the importance of each transition is measured. One idealised criterion would be the amount the RL agent can learn from a transition in its current state (expected learning progress).

その経験から学習したときの更新度合いが優先順位付けのよい指標である。

While this measure is not directly accessible, a reasonable proxy is the magnitude of a transition’s TD errorδ, which indicates how ‘surprising’ or unexpected the transition is: specifically, how far the value is from its next-step bootstrap estimate

が、それは直接利用できないので、TD誤差を代理指標として利用する。TD誤差が大きいということは現在の近似関数が予想外だったということだからである。


greedy-優先度付け：

単にTD誤差の大きいサンプルを優先する。

これは誤差の少ないサンプルが極端に使われなくなること、また環境ノイズ、とくに報酬が確率的な場合に弱いという欠点がある


確率的サンプリング：

サンプルiが選ばれる確率　= サンプルiの優先度^alpha / 総和（サンプルkの優先度^alpha）

サンプルiの優先度の決め方を二つ提案する

1. 線形優先度づけ

優先度 = TD誤差の絶対値

2. ランクにもとづく優先度付け

優先度 = 1 / TD誤差の大きさの順位


SumTreeで実装すると効率的らしい


考察：

rank-baseの方が頑健だと思ってたけど同じくらいだった。たぶんAtari環境では報酬のクリッピングを強く効かせているからだと思う。




For stability reasons, we always normalize weights by 1/maxiwi
so that they only scale the update downwards.

重みが1以上になると(つまり誤差が増幅すると)、安定性が悪くなるので最大値で割ってスケールします
