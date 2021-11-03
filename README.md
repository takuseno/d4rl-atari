![format check](https://github.com/takuseno/d4rl-atari/workflows/format%20check/badge.svg)
![test](https://github.com/takuseno/d4rl-atari/workflows/test/badge.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue)
[![Gitter](https://img.shields.io/gitter/room/d3rlpy/d4rl-atari)](https://gitter.im/d3rlpy/d4rl-atari)

# d4rl-atari
Datasets for Data-Driven Deep Reinforcement Learning with Atari environments.
This project is intending to provide the easy-to-use wrapper for
[the datasets provided by Google](https://research.google/tools/datasets/dqn-replay/).
The dataset structure is designed for Google's [Dopamine](https://github.com/google/dopamine)
so that the datasets are not friendly to developers that don't use the framework.

In order to let everyone use the datasets much easier, this library is designed
as Atari version of [d4rl](https://github.com/rail-berkeley/d4rl).
You can access to the Atari datasets just like d4rl only with few lines of codes.

## install
```
$ pip install git+https://github.com/takuseno/d4rl-atari
```

## usage
The API is mostly identical to the original d4rl.
```py
import gym
import d4rl_atari

env = gym.make('breakout-mixed-v0') # -v{0, 1, 2, 3, 4} for datasets with the other random seeds

# interaction with its environment through dopamine-style Atari wrapper
observation = env.reset() # observation.shape == (84, 84)
observation, reward, terminal, info = env.step(env.action_space.sample())

# dataset will be automatically downloaded into ~/.d4rl/datasets/[GAME]/[INDEX]/[EPOCH]
dataset = env.get_dataset()
dataset['observations'] # observation data in (1000000, 1, 84, 84)
dataset['actions'] # action data in (1000000,)
dataset['rewards'] # reward data in (1000000,)
dataset['terminals'] # terminal flags in (1000000,)
```

The environment produced in this package is wrapped by dopamine-style wrapper.
Thus, the observations between the datasets and the environment are exactly same.
So, if you train your model with the dataset, your model is ready to interact
with the environment.

### stacking observations
You can stack observations.
```py
env = gym.make('breakout-mixed-v0', stack=True)

# stacked gray-scale image
env.reset() # (4, 84, 84)

dataset = env.get_dataset()
dataset['observations'] # a list of 1M arrays with shape of (4, 84, 84)
```

The observations included in dataset are shaped in `(1000000, 84, 84)` without
stacked.
To easily feed this dataset to RL models, the observations should be stacked
with consecutive 4 frames.
However, simply making up ndarray with shape of `(1000000, 4, 84, 84)` consumes
more than 26GiB of memory just for dataset, which is quite large for most
desktop computers.

Therefore, `d4rl-atari` package stacks frames without copying images by
remaining pointers to the original data, which eventually saves around 20GiB
of memory.
To do so, the `dataset['observations']` is a builtin `list` obejct containing
1M 4x84x84 images.

```py
type(dataset['observations']) # list
dataset['observations'][0].shape # (4, 84, 84)
```

## available datasets
You can access to the datasets for all games as long as Google provides it.

```
# GAME should be lower-case letters splitted with '-'
env = gym.make('[GAME]-{mixed,medium,expert}-v{0, 1, 2, 3, 4}')
```

- `mixed` denotes datasets collected at the first 1M steps.
- `medium` denotes datasets collected at between 9M steps and 10M steps.
- `expert` denotes datasets collected at the last 1M steps.

Alternatively, you can get the datasets by specifying an epoch.

```
env = gym.make('[GAME]-epoch-[EPOCH]-v{0, 1, 2, 3, 4}')
```


## contribution
Any contributions will be welcomed!!

### coding style
This repository is formatted with [yapf](https://github.com/google/yapf).
You can format the entire repository follows:
```
$ ./scripts/format
```

## acknowledgement
This work is supported by Information-technology Promotion Agency, Japan
(IPA), Exploratory IT Human Resources Project (MITOU Program) in the fiscal
year 2020.
