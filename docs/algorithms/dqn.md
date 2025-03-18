# DQN


DQN builds upon Q-learning by introducing a replay buffer and target network, key innovations that enhance algorithm stability. For details, see the original paper [Human-level control through deep reinforcement learning ](https://www.nature.com/articles/nature14236). 

## Continuous state - discrete action    

The [```dqn_classical.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_classical.py) and the [```dqn_quantum.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_quantum.py) have the following features:

* ✅ Work with the Box observation space of low-level features
* ✅ Work with the discrete action space
* ✅ Work with envs like [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
* ❌ Multiple Vectorized Environments not enabled
* ❌ No single file implementation (require ```replay_buffer.py```)

### Implementation details

Our implementation of the DQN is essentially the same as in CleanRL. For implementation details of the classical algorithm, we refer to [the CleanRL documentation](https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy). The key difference between the classical and the quantum algorithm is the ```DQNAgentQuantum``` class, as shown below

<div style="display: flex;">
  <span style="width: 50%;">
    ```py title="dqn_quantum.py" linenums="1"
    class DQNAgentQuantum(nn.Module):
        def __init__(self, observation_size, num_actions, config):
            super().__init__()
            self.config = config
            self.observation_size = observation_size
            self.num_actions = num_actions
            self.num_qubits = config["num_qubits"]
            self.num_layers = config["num_layers"]
            # input and output scaling are always initialized as ones
            self.input_scaling = nn.Parameter(
                torch.ones(self.num_layers, self.num_qubits), requires_grad=True
            )
            self.output_scaling = nn.Parameter(
                torch.ones(self.num_actions), requires_grad=True
            )
            # trainable weights are initialized randomly between -pi and pi
            self.weights = nn.Parameter(
                torch.FloatTensor(self.num_layers, self.num_qubits*2)
                .uniform_(-np.pi, np.pi),
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
    ```py title="dqn_classical.py" linenums="1"
    class DQNAgentClassical(nn.Module):
        def __init__(self, observation_size, num_actions):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(observation_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions),
            )
    ```
  </span>
</div>

<div style="display: flex;">
  <span style="width: 50%;">
    ```py title="dqn_quantum.py"
    def forward(self, x):
        logits = self.quantum_circuit(
            x,
            self.input_scaling,
            self.weights,
            self.num_qubits,
            self.num_layers,
            self.num_actions,
            self.observation_size
        )
        logits = torch.stack(logits, dim=1)
        logits = logits * self.output_scaling
        return logits
    ```
  </span>
 <span style="width: 50%;">
    ```py title="dqn_classical.py"
    def forward(self, x):
        return self.network(x)
    ```
  </span>
</div>

Additionally, to these changes to the ```Agent```class, we also need to specify a function for the ansatz of the parameterized quantum circuit. 

```py title="dqn_quantum.py" linenums="1"
def parameterized_quantum_circuit(
    x, input_scaling, weights, num_qubits, num_layers, num_actions, observation_size
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

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_actions)]
```

The ansatz of this parameterized quantum circuit is taken from the publication by Skolik et al [Quantum agents in the Gym](https://quantum-journal.org/papers/q-2022-05-24-720/pdf/). The ansatz is also depicted in the figure below:

Our implementation hence incorporates some key novelties proposed by Skolik:

* ```data reuploading```: In our ansatz, the features of the states are encoded via RX rotation gates. Instead of only encoding the features in the first layer, this process is repeated in each layer. This has shown to improve training performance.
* ```input scaling```: In our implementation, we define additionally to the trainable weights of the
* ```output scaling```: In our implementation, we define additionally to the trainable weights of the  

We also provide the option to select different ```learning rates``` for the different parameter sets:

```py title="dqn_quantum.py"
    optimizer = optim.Adam(
        [
            {"params": agent.input_scaling, "lr": lr_input_scaling},
            {"params": agent.output_scaling, "lr": lr_output_scaling},
            {"params": agent.weights, "lr": lr_weights},
        ]
    )
```

Also, you can use a faster pennylane backend for your simulations:

* ```pennylane-lightning```: We enable the use of the ```lightning``` simulation backend by pennylane, which speeds up simulation 

We also add an observation wrapper called ```ArctanNormalizationWrapper``` at the very beginning of the file. Because we encode the features of the states as rotations, we need to ensure that the features are not beyond the interval of - π and π due to the periodicity of the rotation gates. For more details on wrappers, see [Advanced Usage](https://georgkruse.github.io/cleanqrl-docs/advanced_usage/jumanji_environments/).

### Experiment results

Next to the 

## Discrete state - discrete action    

The [```dqn_classical_discrete_state.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_classical_discrete_state.py) and the [```dqn_quantum_discrete_state.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_quantum_discrete_state.py) have the following features:

* ✅ Work with the discrete observation space 
* ✅ Work with the discrete action space
* ✅ Work with envs like [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
* ❌ Multiple Vectorized Environments not enabled
* ❌ No single file implementation (require ```replay_buffer.py```)

### Implementation details

The implementations follow the same principles as [```dqn_classical.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_classical.py) and [```dqn_quantum.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_quantum.py). In the following we focus on the key differences between [```dqn_quantum_discrete_state.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_quantum_discrete_state.py) and [```dqn_quantum.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_quantum.py). The same differences also apply for the classical implementations.

The key difference is the state encoding. Since discrete state environments like FrozenLake return an integer value, it is straight forward to encode the state as a binary value instead of an integer. For that, an additional function for ```DQNAgentQuantuM``` is added called ```encoding_input```. This converts the integer value into its binary value.

```py title="dqn_quantum_discrete_state.py"
    def encode_input(self, x):
        x_binary = torch.zeros((x.shape[0], self.observation_size))
        for i, val in enumerate(x):
            binary = bin(int(val.item()))[2:]
            padded = binary.zfill(self.observation_size)
            x_binary[i] = torch.tensor([int(bit) * np.pi for bit in padded])
        return x_binary
```
Now we just need to also call this function before we pass the input to the parameterized quantum circuit:

```py title="dqn_quantum_discrete_state.py" hl_lines="2 4"
    def forward(self, x):
        x_encoded = self.encode_input(x)
        logits = self.quantum_circuit(
            x_encoded,
            self.input_scaling,
            self.weights,
            self.num_qubits,
            self.num_layers,
            self.num_actions,
            self.observation_size
        )
        logits = torch.stack(logits, dim=1)
        logits = logits * self.output_scaling
        return logits

```
### Experiment results


## Jumanji Environments    

The [```dqn_classical_jumanji.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_classical_jumanji.py) and the [```dqn_quantum_jumanji.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_quantum_jumanji.py) have the following features:

* ✅ Work with [jumanji](https://github.com/instadeepai/jumanji) environments 
* ✅ Work with envs like [Traveling Salesperson](https://instadeepai.github.io/jumanji/environments/tsp/) and [Knapsack](https://instadeepai.github.io/jumanji/environments/knapsack/)
* ❌ Multiple Vectorized Environments not enabled
* ❌ No single file implementation (require ```replay_buffer.py```)
* ❌ Require custom wrapper file for jumanji (```jumanji_wrapper.py```)

### Implementation details

The implementations follow the same principles as [```dqn_classical.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_classical.py) and [```dqn_quantum.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_quantum.py). In the following we focus on the key differences between [```dqn_quantum_jumanji.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_quantum_jumanji.py) and [```dqn_quantum.py```](https://github.com/georgkruse/cleanqrl/blob/main/cleanqrl/dqn_quantum.py). The same differences also apply for the classical implementations.

For most of the ```jumanji`` environments, the observation space is quite complex. Instead of simple numpy arrays for the states, we often have dictionary states which vary in size and shape. E.g. the Knapsack problem returns a state of shape 

* ```weights```: jax array (float) of shape (num_items,), array of weights of the items to be packed into the knapsack.
* ```values```: jax array (float) of shape (num_items,), array of values of the items to be packed into the knapsack.
* ```packed_items```: jax array (bool) of shape (num_items,), array of binary values denoting which items are already packed into the knapsack.
* ```action_mask```: jax array (bool) of shape (num_items,), array of binary values denoting which items can be packed into the knapsack.

In order to parse this to a parameterized quantum circuit or a neural network, we can use a gym wrapper which converters the state again to an array. This is being done when calling the function ```create_jumanji_wrapper```. For more details see [Jumanji Wrapper](https://georgkruse.github.io/cleanqrl-docs/advanced_usage/jumanji_environments/). 

```py title="dqn_quantum_jumanji.py" hl_lines="3"

def make_env(env_id, config):
    def thunk():
        env = create_jumanji_env(env_id, config)
        env = ReplayBufferWrapper(env)

        return env

    return thunk
```

```py title="dqn_quantum_jumanji.py" hl_lines="2 8"
class DQNAgentQuantum(nn.Module):
    def __init__(self, num_actions, config):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.num_qubits = config["num_qubits"]
        self.num_layers = config["num_layers"]
        self.block_size = 3  # number of subblocks depends on environment
```

Because the state is now growing quickly in size as the number of items of the Knapsack is increased, we use a slightly different approach to encode it: Instead of encoding each feature of the state on an individual qubit, we divide the state again into 3 blocks, namely ```weights```, ```values```and ```packed_items```. This will be important for the modification of our ansatz for the parameterized quantum circuit which you can see below:

```py title="dqn_quantum_jumanji.py" hl_lines="9-11 14-22"
def parametrized_quantum_circuit(
    x, input_scaling, weights, num_qubits, num_layers, num_actions
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

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_actions)]
```

We encode each of these blocks individually in each layer. By that, we can save quantum circuit width, so the number of qubits, by increasing our quantum circuit depth, so the number of gates we are using. See our [**Tutorials**](https://georgkruse.github.io/cleanqrl-docs/tutorials/overview/) section for better ansatz design.

### Experiment results

