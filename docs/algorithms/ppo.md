# PPO
https://arxiv.org/pdf/1509.02971.pdf)
[Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) has become a leading algorithm in deep reinforcement learning due to its robust performance and relative simplicity. It can utilize parallel environments for faster training and supports diverse action spaces, enabling its application to a wide variety of tasks, including many games. PPO achieves stability by limiting the policy update at each step, preventing drastic changes that can derail learning – a key advantage over earlier policy gradient methods. Furthermore, it exhibits better sample efficiency than algorithms like DQN. Our implementation mainly follows the one provided by [CleanRL](https://docs.cleanrl.dev/rl-algorithms/ppo/).

## Continuous state - discrete action    

The [```ppo_classical.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical.py) and the [```ppo_quantum.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum.py) have the following features:

* ✅ Work with the Box observation space of low-level features
* ✅ Work with the discrete action space
* ✅ Work with envs like [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
* ✅ Multiple Vectorized Environments 
* ✅ Single file implementation 

### Implementation details

The key difference between the classical and the quantum algorithm is the ```PPOAgentQuantum``` class, as shown below

<div style="display: flex;">
  <span style="width: 50%;">
    ```py title="ppo_quantum.py" linenums="1"
    class PPOAgentQuantum(nn.Module):
        def __init__(self, observation_size, num_actions, config):
            super().__init__()
            self.config = config
            self.observation_size = observation_size
            self.num_actions = num_actions
            self.num_qubits = config["num_qubits"]
            self.num_layers = config["num_layers"]

            # input and output scaling are always initialized as ones
            self.input_scaling_critic = nn.Parameter(
                torch.ones(self.num_layers, self.num_qubits), requires_grad=True
            )
            self.output_scaling_critic = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            # trainable weights are initialized randomly between -pi and pi
            self.weights_critic = nn.Parameter(
                torch.FloatTensor(self.num_layers, self.num_qubits * 2).uniform_(
                    -np.pi, np.pi
                ),
                requires_grad=True,
            )

            # input and output scaling are always initialized as ones
            self.input_scaling_actor = nn.Parameter(
                torch.ones(self.num_layers, self.num_qubits), requires_grad=True
            )
            self.output_scaling_actor = nn.Parameter(
                torch.ones(self.num_actions), requires_grad=True
            )
            # trainable weights are initialized randomly between -pi and pi
            self.weights_actor = nn.Parameter(
                torch.FloatTensor(self.num_layers, self.num_qubits * 2).uniform_(
                    -np.pi, np.pi
                ),
                requires_grad=True,
            )

            device = qml.device(config["device"], wires=range(self.num_qubits))
            self.quantum_circuit = qml.QNode(
                parameterized_quantum_circuit,
                device,
                diff_method=config["diff_method"],
                interface="torch",
            )
    ```
  </span>
  <span style="width: 51%;">
    ```py title="ppo_classical.py" linenums="1"
    class PPOAgentClassical(nn.Module):
        def __init__(self, envs):
            super().__init__()
            self.critic = nn.Sequential(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.actor = nn.Sequential(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, envs.single_action_space.n),
            )
    ```
  </span>
</div>

<div style="display: flex;">
  <span style="width: 50%;">
    ```py title="ppo_quantum.py"
    def get_value(self, x):
        value = self.quantum_circuit(
            x,
            self.input_scaling_critic,
            self.weights_critic,
            self.num_qubits,
            self.num_layers,
            self.num_actions,
            self.observation_size,
            "critic",
        )
        value = torch.stack(value, dim=1)
        value = value * self.output_scaling_critic
        return value

    def get_action_and_value(self, x, action=None):
        logits = self.quantum_circuit(
            x,
            self.input_scaling_actor,
            self.weights_actor,
            self.num_qubits,
            self.num_layers,
            self.num_actions,
            self.observation_size,
            "actor",
        )
        logits = torch.stack(logits, dim=1)
        logits = logits * self.output_scaling_actor
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)
    ```
  </span>
 <span style="width: 50%;">
    ```py title="ppo_classical.py"
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    ```
  </span>
</div>

Additionally to these changes to the ```Agent```class, we also need to specify a function for the ansatz of the parameterized quantum circuit. We can reuse most of the circuit by passing an additional parameter which we will call ```agent_type```. By doing so, we ensure that we either return a single expectation value for the critic or a tensor of expectation values of the shape ```num_actions```:

```py title="ppo_quantum.py" linenums="1"
def parameterized_quantum_circuit(
    x, input_scaling, weights, num_qubits, num_layers, num_actions, observation_size, agent_type
):
    for layer in range(num_layers):
        for i in range(observation_size):
            qml.RX(input_scaling[layer, i] * x[:, i], wires=[i])

        for i in range(num_qubits):
            qml.RY(weights[layer, i], wires=[i])

        for i in range(num_qubits):
            qml.RZ(weights[layer, i + num_qubits], wires=[i])

        if num_qubits == 2:
            qml.CZ(wires=[0, 1])
        else:
            for i in range(num_qubits):
                qml.CZ(wires=[i, (i + 1) % num_qubits])

    if agent_type == "actor":
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_actions)]
    elif agent_type == "critic":
        return [qml.expval(qml.PauliZ(0))]

```

The ansatz of this parameterized quantum circuit is taken from the publication of by Skolik et al [Quantum agents in the Gym](https://quantum-journal.org/papers/q-2022-05-24-720/pdf/). The ansatz is also depicted in the figure below:

Our implementation hence build open the work by Jerbi but also incorporates some key novelties proposed by Skolik:

* ```data reuploading```: In our ansatz, the features of the states are encoded via RX rotation gates. Instead of only encoding the features in the first layer, this process is repeated in each layer. This has shown to improve training performance.
* ```input scaling```: In our implementation, we define additionally to the trainable weights of the
* ```output scaling```: In our implementation, we define additionally to the trainable weights of the  

We also provide the option to select different ```learning rates``` for the different parameter sets for the actor and the critic:

```py title="ppo_quantum.py"
    optimizer = optim.Adam(
        [
            {"params": agent.input_scaling_actor, "lr": lr_input_scaling},
            {"params": agent.output_scaling_actor, "lr": lr_output_scaling},
            {"params": agent.weights_actor, "lr": lr_weights},
            {"params": agent.input_scaling_critic, "lr": lr_input_scaling},
            {"params": agent.output_scaling_critic, "lr": lr_output_scaling},
            {"params": agent.weights_critic, "lr": lr_weights},
        ]
    )
```

Also, you can use a faster pennylane backend for your simulations:

* ```pennylane-lightning```: We enable the use of the ```lightning``` simulation backend by pennylane, which speeds up simulation 

We also add an observation wrapper called ```ArctanNormalizationWrapper``` at the very beginning of the file. Because we encode the features of the states as rotations, we need to ensure that the features are not beyond the interval of - π and π due to the periodicity of the rotation gates. For more details on wrappers, see [Advanced Usage](https://georgkruse.github.io/cleanqrl-docs/advanced_usage/jumanji_environments/).


### Experiment results


## Continuous state - continuous action    

The [```ppo_classical_continuous_action.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical_continuous_action.py) and the [```ppo_quantum_continuous_action.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum_continuous_action.py) have the following features:

* ✅ Work with the continuous observation space 
* ✅ Work with the continuous action space
* ✅ Work with envs like [Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/)
* ✅ Multiple Vectorized Environments 
* ✅ Single file implementation 

### Implementation details

The implementations follow the same principles as [```ppo_classical.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical.py) and [```ppo_quantum.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum.py). In the following we focus on the key differences between [```ppo_quantum_continuous_action.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum_continuous_action.py) and [```ppo_quantum.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum_continuous_action.py). The same differences also apply for the classical implementations.

While the state encoding is the same as for the previous approach, we need to implement some modifications in order to draw continuous actions with the parameterized quantum circuit. For that we modify the ```PPOAgentQuantum``` class as follows:

```py title="ppo_quantum_continuous_action.py" hl_lines="14 17-31"

class PPOAgentQuantumContinuous(nn.Module):
    def __init__(self, observation_size, num_actions, config):
        super().__init__()
        self.config = config
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]

        .....


        # additional trainable parameters for the variance of the continuous actions
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))

    def get_action_and_value(self, x, action=None):
        action_mean = self.quantum_circuit(
            x,
            self.input_scaling_actor,
            self.weights_actor,
            self.num_qubits,
            self.num_layers,
            self.num_actions,
            self.observation_size,
            "actor",
        )
        action_mean = torch.stack(action_mean, dim=1)
        action_mean = action_mean * self.output_scaling_actor
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.get_value(x),
        )
```

In our implementation, the mean of the continuous action is based on the expectation value of the parameterized quantum circuit, while the variance is an additional classical trainable parameter. This parameter is also the same for all continuous actions. For additional information we refer to [Variational Quantum Circuit Design for Quantum Reinforcement Learning on Continuous Environments](https://arxiv.org/pdf/2312.13798).


### Experiment results


## Discrete state - discrete action    

The [```ppo_classical_discrete_state.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical_discrete_state.py) and the [```ppo_quantum_discrete_state.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum_discrete_state.py) have the following features:

* ✅ Work with the discrete observation space 
* ✅ Work with the discrete action space
* ✅ Work with envs like [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
* ✅ Multiple Vectorized Environments 
* ✅ Single file implementation 

### Implementation details

The implementations follow the same principles as [```ppo_classical.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical.py) and [```ppo_quantum.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum.py). In the following we focus on the key differences between [```ppo_quantum_discrete_state.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum_discrete_state.py) and [```ppo_quantum.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum.py). The same differences also apply for the classical implementations.

The key difference is the state encoding. Since discrete state environments like FrozenLake return an integer value, it is straight forward to encode the state as a binary value instead of an integer. For that, an additional function for ```PPOAgentQuantum``` is added called ```encoding_input```. This converts the integer value into its binary value.

```py title="ppo_quantum_discrete_state.py"
    def encode_input(self, x):
        x_binary = torch.zeros((x.shape[0], self.observation_size))
        for i, val in enumerate(x):
            binary = bin(int(val.item()))[2:]
            padded = binary.zfill(self.observation_size)
            x_binary[i] = torch.tensor([int(bit) * np.pi for bit in padded])
        return x_binary
```
Now we just need to also call this function before we pass the input to the parameterized quantum circuit:

```py title="ppo_quantum_discrete_state.py" hl_lines="2 4"
    def get_action_and_value(self, x, action=None):
        x_encoded = self.encode_input(x)
        logits = self.quantum_circuit(
            x_encoded,
            self.input_scaling_actor,
            self.weights_actor,
            self.num_qubits,
            self.num_layers,
            self.num_actions,
            self.observation_size,
            "actor",
        )
        logits = torch.stack(logits, dim=1)
        logits = logits * self.output_scaling_actor
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)

```
### Experiment results


## Jumanji Environments    

The [```ppo_classical_jumanji.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical_jumanji.py) and the [```ppo_quantum_jumanji.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum_jumanji.py) have the following features:

* ✅ Work with [jumanji](https://github.com/instadeepai/jumanji) environments 
* ✅ Work with envs like [Traveling Salesperson](https://instadeepai.github.io/jumanji/environments/tsp/) and [Knapsack](https://instadeepai.github.io/jumanji/environments/knapsack/)
* ✅ Multiple Vectorized Environments 
* ❌ No single file implementation (require custom wrapper file for jumanji ```jumanji_wrapper.py```)

### Implementation details

The implementations follow the same principles as [```ppo_classical.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_classical.py) and [```ppo_quantum.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum.py). In the following we focus on the key differences between [```ppo_quantum_jumanji.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum_jumanji.py) and [```ppo_quantum.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/ppo_quantum.py). The same differences also apply for the classical implementations.

For most of the ```jumanji`` environments, the observation space is quite complex. Instead of simple numpy arrays for the states, we often have dictionary states which vary in size and shape. E.g. the Knapsack problem returns a state of shape 

* ```weights```: jax array (float) of shape (num_items,), array of weights of the items to be packed into the knapsack.
* ```values```: jax array (float) of shape (num_items,), array of values of the items to be packed into the knapsack.
* ```packed_items```: jax array (bool) of shape (num_items,), array of binary values denoting which items are already packed into the knapsack.
* ```action_mask```: jax array (bool) of shape (num_items,), array of binary values denoting which items can be packed into the knapsack.

In order to parse this to a parameterized quantum circuit or a neural network, we can use a gym wrapper which converters the state again to an array. This is being done when calling the function ```create_jumanji_wrapper```. For more details see [Jumanji Wrapper](https://georgkruse.github.io/cleanqrl-docs/advanced_usage/jumanji_environments/). 

```py title="ppo_quantum_jumanji.py" hl_lines="3"

def make_env(env_id, config):
    def thunk():
        env = create_jumanji_env(env_id, config)

        return env

    return thunk
```

```py title="ppo_quantum_jumanji.py" hl_lines="2 8"
class PPOAgentQuantum(nn.Module):
    def __init__(self, num_actions, config):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]
        self.block_size = 3  # number of subblocks depends on environment
```

Because the state is now growing quickly in size as the number of items of the Knapsack is increased, we use a slightly different approach to encode it: Instead of encoding each feature of the state on an individual qubit, we divide the state again into 3 blocks, namely ```weights```, ```values```and ```packed_items```. This will be important for the modification of our ansatz for the parameterized quantum circuit which you can see below:

```py title="ppo_quantum_jumanji.py" hl_lines="9-11 14-22"
def parametrized_quantum_circuit(
    x, input_scaling, weights, num_qubits, num_layers, num_actions, agent_type
):

    # This block needs to be adapted depending on the environment.
    # The input vector is of shape [4*num_actions] for the Knapsack:
    # [action mask, packed items, values, weights]

    annotations = x[:, num_qubits : num_qubits * 2]
    values_kp = x[:, num_qubits * 2 : num_qubits * 3]
    weights_kp = x[:, 3*num_qubits:]

    for layer in range(num_layers):
        for block, features in enumerate([annotations, values_kp, weights_kp]):
            for i in range(num_qubits):
                qml.RX(input_scaling[layer, block, i] * features[:, i], wires=[i])

            for i in range(num_qubits):
                qml.RY(weights[layer, block, i], wires=[i])

            for i in range(num_qubits):
                qml.RZ(weights[layer, block, i+num_qubits], wires=[i])
        
            if num_qubits == 2:
                qml.CZ(wires=[0, 1])
            else:
                for i in range(num_qubits):
                    qml.CZ(wires=[i, (i + 1) % num_qubits])

    if agent_type == "actor":
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_actions)]
    elif agent_type == "critic":
        return [qml.expval(qml.PauliZ(0))]
```

We encode each of these blocks individually in each layer. By that, we can save quantum circuit width, so the number of qubits, by increasing our quantum circuit depth, so the number of gates we are using. However, this still is not an optimal encoding. See our [**Tutorials**](https://georgkruse.github.io/cleanqrl-docs/tutorials/overview/) section for better ansatz design.

### Experiment results

