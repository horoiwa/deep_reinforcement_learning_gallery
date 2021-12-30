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


rewardの予測は正規分布のlogProbを最大化
discountの予測はベルヌーイ分布のLogProbを最大化


For Atari, we find Reinforce gradients to work substantially better and use ρ = 1 and η = 10−3
. For
continuous control, we find dynamics backpropagation to work substantially better and use ρ = 0
and η = 10−4
. Annealing these hyper parameters can improve performance slightly but to avoid the
added complexity we report the scores without anneali


```
{'logdir': './logdir', 'seed': 0, 'task': 'atari_pong', 'envs': 1, 'envs_parallel': 'none', 'render_size': (64, 64), 'dmc_camera': -1, 'atari_grayscale': True, 'time_limit': 100, 'action_repeat': 4, 'steps': 50000000.0, 'log_every': 300.0, 'eval_every': 300.0, 'eval_eps': 1, 'prefill': 100, 'pretrain': 1, 'train_every': 16, 'train_steps': 1, 'expl_until': 0, 'replay': {'capacity': 2000000.0, 'ongoing': False, 'minlen': 10, 'maxlen': 30, 'prioritize_ends': True}, 'dataset': {'batch': 10, 'length': 10}, 'log_keys_video': ('image',), 'log_keys_sum': '^$', 'log_keys_mean': '^$', 'log_keys_max': '^$', 'precision': 16, 'jit': False, 'clip_rewards': 'tanh', 'expl_behavior': 'greedy', 'expl_noise': 0.0, 'eval_noise': 0.0, 'eval_state_mean': False, 'grad_heads': ('decoder', 'reward', 'discount'), 'pred_discount': True, 'rssm': {'ensemble': 1, 'hidden': 600, 'deter': 600, 'stoch': 32, 'discrete': 32, 'act': 'elu', 'norm': 'none', 'std_act': 'sigmoid2', 'min_std': 0.1}, 'encoder': {'mlp_keys': '$^', 'cnn_keys': 'image', 'act': 'elu', 'norm': 'none', 'cnn_depth': 48, 'cnn_kernels': (4, 4, 4, 4), 'mlp_layers': (400, 400, 400, 400)}, 'decoder': {'mlp_keys': '$^', 'cnn_keys': 'image', 'act': 'elu', 'norm': 'none', 'cnn_depth': 48, 'cnn_kernels': (5, 5, 6, 6), 'mlp_layers': (400, 400, 400, 400)}, 'reward_head': {'layers': 4, 'units': 400, 'act': 'elu', 'norm': 'none', 'dist': 'mse'}, 'discount_head': {'layers': 4, 'units': 400, 'act': 'elu', 'norm': 'none', 'dist': 'binary'}, 'loss_scales': {'kl': 0.1, 'reward': 1.0, 'discount': 5.0, 'proprio': 1.0}, 'kl': {'free': 0.0, 'forward': False, 'balance': 0.8, 'free_avg': True}, 'model_opt': {'opt': 'adam', 'lr': 0.0002, 'eps': 1e-05, 'clip': 100, 'wd': 1e-06}, 'actor': {'layers': 4, 'units': 400, 'act': 'elu', 'norm': 'none', 'dist': 'onehot', 'min_std': 0.1}, 'critic': {'layers': 4, 'units': 400, 'act': 'elu', 'norm': 'none', 'dist': 'mse'}, 'actor_opt': {'opt': 'adam', 'lr': 4e-05, 'eps': 1e-05, 'clip': 100, 'wd': 1e-06}, 'critic_opt': {'opt': 'adam', 'lr': 0.0001, 'eps': 1e-05, 'clip': 100, 'wd': 1e-06}, 'discount': 0.999, 'discount_lambda': 0.95, 'imag_horizon': 15, 'actor_grad': 'reinforce', 'actor_grad_mix': 0.1, 'actor_ent': 0.001, 'slow_target': True, 'slow_target_update': 100, 'slow_target_fraction': 1, 'slow_baseline': True, 'reward_norm': {'momentum': 1.0, 'scale': 1.0, 'eps': 1e-08}, 'expl_intr_scale': 1.0, 'expl_extr_scale': 0.0, 'expl_opt': {'opt': 'adam', 'lr': 0.0003, 'eps': 1e-05, 'clip': 100, 'wd': 1e-06}, 'expl_head': {'layers': 4, 'units': 400, 'act': 'elu', 'norm': 'none', 'dist': 'mse'}, 'expl_reward_norm': {'momentum': 1.0, 'scale': 1.0, 'eps': 1e-08}, 'disag_target': 'stoch', 'disag_log': False, 'disag_models': 10, 'disag_offset': 1, 'disag_action_cond': True, 'expl_model_loss': 'kl'}
```
