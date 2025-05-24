# EfficientZeroV2

This is **unofficial** **simplificated** re-implementation of (EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data)[https://github.com/Shengjiewang-Jason/EfficientZeroV2]


## Simplifications

- Simplifications due to limited compute resource
  - Smaller batch size
  - Replaced Batch Normalization with Group Normalization due to smaller batch size
  - fewer number of gradient steps
  - fewer number of simulations
- No Prioritized Experience Replay
- No value-prefix prediction, use simple reward prediction on each time step without LSTM.
- No mixed value approximation
- Use grayscale instead of RGB as input


## Requirements
`pip install tensorflow==2.17.0 numpy==1.24.4 "gymnasium[atari, accept-rom-license]"`


