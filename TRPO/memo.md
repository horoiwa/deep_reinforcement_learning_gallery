## TRPO

信頼領域ポリシー最適化

walker, hopping, atari

ハイパラチューニングがあまりいらない

発展： PPO ACKTR


まずはPolicy Gradientの証明：
毎回アドバンテージ最大化するようにアップデートすることでポリシーが改善する


PGは方向を決めてから更新幅にそって直線的に更新する

これでは容易に足を踏み外してしまう

下限を近似する
MinMax戦略

自然勾配
https://towardsdatascience.com/natural-gradient-ce454b3dcdfa

https://medium.com/@jonathan_hui/rl-trust-region-policy-optimization-trpo-explained-a6ee04eeeee9

https://drive.google.com/file/d/0BxXI_RttTZAhMVhsNk5VSXU0U3c/view

著者の解説だった
http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_13_advanced_pg.pdf


HVPの実装
https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/

テイラー展開が詳しい

http://www.andrew.cmu.edu/course/10-703/slides/Lecture_NaturalPolicyGradientsTRPOPPO.pdf

https://www.youtube.com/watch?v=xvRrgxcpaHY

- MMアルゴリズム

上界最小化アルゴリズム

もとの方策関数から離れることにペナルティを与える

- 重要度サンプリング

Lはadvantageの重要度サンプリング

- 信頼領域


パラメータが近い != ポリシーの出力が近い なので学習率小さくすればいいじゃんとかAdamアプローチは本質的には無効


oldポリシーのアドバンテージで新しいポリシーのパフォーマンスを評価できる
でもs,aの分布は新しいポリシーに従う


目的関数

重要度サンプリング

(1-γ)の証明

割引率０で考える

s' = s はポリシーが十分近ければ成立する

十分がどのくらいかも証明されている


A3Cとならぶ強化学習初学者の鬼門であるTRPO

theta_old周辺ならPGと一致する


L(threta) - C KL(pi , piold)
※Cはハイパラではない

が常に正しいと証明されているが, まじめにやるとステップサイズが小さくなりすぎるので制約つき最適化問題をとく


ヘッセ行列は計算量ヤバい


Hx = g = grad(sum(x.dot(grad(KL))))
x = H-1g

https://roosephu.github.io/2016/11/19/TRPO/#:~:text=Fisher%2Dvector%20product,%CE%B8)%E2%8A%A4%7C%CE%B8%5D.


F ≒ hessian-1 と近似している


This mathematical principled method to compute the step size and direction is the major contribution of TRPO. Compare this with the ad hoc “learning rate schedule” typically used in training neural networks.
