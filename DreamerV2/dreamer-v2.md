https://openreview.net/forum?id=0oabwyZbOu

- 公式実装

https://github.com/danijar/dreamerv2

- google AI blog

https://ai.googleblog.com/2021/02/mastering-atari-with-discrete-world.html

https://ai.googleblog.com/2020/03/introducing-dreamer-scalable.html

- Project Web Site

https://danijar.com/project/dreamerv2/

- Article

https://arxiv.org/pdf/2010.02193.pdf

https://arxiv.org/pdf/1912.01603.pdf


復元可能な潜在変数空間上での未来予測

- 世界モデルはサンプル効率が（ほんとうは）よいはず
- 世界モデルの構築はオフラインデータと相性がよい
- 微分可能なシミュレータ



コンパクト表現を学習

RainbowとIQNの最終的なパフォーマンスを上回っています


言葉遊びだけどmujoco環境はモデルベース。待ったを許可するならモデルベース


- メトリックの改善を主張

正直Dreamer関係ない。
指標の改善は既存SotAを達成したときに言わないと逃げみたいになって説得力持ちにくいよね。
分野の権威から言い出すのも大事。

メジアンは比較的ましだがモンテスマを無視するし、平均はスコアが青天井ゲームに引きずられる

- latent variables

- KL バランシング

- image grad


未来予測性能を再構築ロスで計る

でも即時報酬予測性能で計ってもよいはず


ざっくりした定義。古くからある

- Reconstruction

- reward 予測
- contrastive estimation


Mastering XXというタイトルはDM的にかなり意味深い


完全にWorld Model(教師あり学習)だけでValueとPolicyを学習する

→ オフラインセッティング


V2では微分可能であることを生かさず（DDPG的な）、REINFORCE(方策勾配)

Reinforce only Reinforce gradients worked substantially better for Atari than dynamics backpropagation. For continuous control, dynamics backpropagation worked substantially better.


discountは最初と最後だけ異なる

## Memo

```
(Pdb) self.shapes
DictWrapper({'image': (64, 64, 1), 'ram': (128,), 'reward': (), 'is_first': (), 'is_last': (), 'is_terminal': ()})

  state = dict(
      #: カテゴリ分布 logit
      logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
      #: カテゴリ分布 softmax?
      stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
      #: GRU initial state
      deter=self._cell.get_initial_state(None, batch_size, dtype))
```

ramは128バイト
