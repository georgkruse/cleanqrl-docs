# Algorithms

## Overview


| Algorithm      | Variants Implemented |
| ----------- | ----------- |
| ✅ [REINFORCE]() |  [`reinforce_classical.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/reinforce_classical.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#reinforce_classicalpy) |
| | [`reinforce_quantum.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/reinforce_quantum.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#reinforce_quantumpy) |
| ✅ [Deep Q-Learning (DQN)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) | [`dqn_classical.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_classical.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#dqn_classicalpy) |
| | [`dqn_quantum.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_quantum.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#dqn_quantumpy) |
| ✅ [Proximal Policy Gradient (PPO)](https://arxiv.org/pdf/1707.06347.pdf)  |  [`ppo_classical.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ppo_classicalpy) |
| |  [`ppo_classical_continuous.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical_continuous.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ppo_classical_continuouspy) |
| |  [`ppo_classical_jumanji.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical_jumanji.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ppo_classical_jumanjipy) |
| |  [`ppo_quantum.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ppo_quantumpy) |
| |  [`ppo_quantum_continuous.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum_continuous.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ppo_quantum_continuouspy) |
| |  [`ppo_quantum_jumanji.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum_jumanji.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ppo_quantum_jumanjipy) |
| ✅ [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) |  [`ddpg_classical.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ddpg_classical.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ddpg_classicalpy) |
| | [`ddpg_quantum.py`](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ddpg_quantum.py), [docs](https://georgkruse.github.io/cleanqrl-docs/algorithms/#ddpg_quantumpy) |


## REINFORCE

### ```reinforce_classical.py```

The ```reinforce_classical.py``` has the following features:

* ✅ Works with the Box observation space of low-level features
* ✅ Works with the Discrete action space
* ✅ Works with envs like CartPole-v1
* ✅ Vectorized Environments 

#### Implementation details

#### Experiment results

### ```reinforce_quantum.py```

The ```reinforce_quantum.py``` has the following features:

* ✅ Works with the Box observation space of low-level features
* ✅ Works with the Discrete action space
* ✅ Works with envs like CartPole-v1
* ✅ Vectorized Environments 

#### Implementation details

#### Experiment results


## DQN

### ```dqn_classical.py```

The ```dqn_classical.py``` has the following features:

* ✅ Works with the Box observation space of low-level features
* ✅ Works with the Discrete action space
* ✅ Works with envs like CartPole-v1
* ❌ Vectorized Environments not enabled
* ❌ Requieres ```replay_buffer.py``` (no single file implementation)

#### Implementation details

#### Experiment results

### ```dqn_quantum.py```

The ```dqn_quantum.py``` has the following features:

* ✅ Works with the Box observation space of low-level features
* ✅ Works with the Discrete action space
* ✅ Works with envs like CartPole-v1
* ❌ Vectorized Environments not enabled
* ❌ Requieres ```replay_buffer.py``` (no single file implementation)

#### Implementation details

#### Experiment results