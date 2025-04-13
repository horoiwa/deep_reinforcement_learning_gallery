# EfficientZeroV2

This is **unofficial** **simplified** re-implementation of (EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data)[https://github.com/Shengjiewang-Jason/EfficientZeroV2]


## Simplifications

- Use grayscale instead of RGB as input
- Use Deterministic-v4 env step instead of NoFrameSKip-v4
- No Prioritized Experience Replay
- No value-prefix for reward prediction, use simple reward prediction on each time step
- No LSTM for reward prediction
- No mixed value target, only use search-based value estimation target (SVE target)
- No mcts-improved-policy based on completed Q, instead, use simple loss

