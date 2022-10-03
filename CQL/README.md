# Conservative Q Learning

https://sites.google.com/view/cql-offline-rl


## Download DQN-Replay dataset

https://research.google/tools/datasets/dqn-replay/

https://offline-rl.github.io/

```
mkdir dqn-replay-dataset && cd ./dqn-replay-dataset
gsutil -m cp -R gs://atari-replay-datasets/dqn/BreakOut .
```

## Dopamine-rl Required

`pip install dopamine-rl`
