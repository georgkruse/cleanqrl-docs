# Get Started

## Installation

To run experiments locally, you need to clone the repository and install a python environment.

```bash
git clone https://github.com/georgkruse/cleanqrl.git
cd cleanqrl
conda env create -f environment.yaml
```

That's it, now you're set up!

## Run first experiments

Each agent can be run as a single file, either from the parent directory or directly in the subfolder. First, activate the environment ```cleanqrl``` and then execute the algorithm's python file:

```
conda activate cleanqrl
python cleanrl/reinforce_quantum.py 
```

or go directly into the folder and execute

```
conda activate cleanqrl
cd cleanqrl 
python reinforce_quantum.py 
```

Before you execute the files, customize the parameters in the  ```Config``` class at the end of each file. Every file has such a dataclass object and the algorithm is callable as a function which takes the config as input:


```python
def reinforce_quantum(config):
    num_envs = config["num_envs"]
    total_timesteps = config["total_timesteps"]
    env_id = config["env_id"]
    gamma = config["gamma"]
    lr_input_scaling = config["lr_input_scaling"]
    lr_weights = config["lr_weights"]
    lr_output_scaling = config["lr_output_scaling"]
    .... 
```

This function can also be called from an external file (see below for details). But first, lets have a closer look to the the ```Config```: 

```py title="reinforce_quantum.py"
@dataclass
class Config:
    # General parameters
    trial_name: str = 'reinforce_quantum'  # Name of the trial
    trial_path: str = 'logs'  # Path to save logs relative to the parent directory
    wandb: bool = False # Use wandb to log experiment data 
    project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project

    # Environment parameters
    env_id: str = "CartPole-v1" # Environment ID
    
    # Algorithm parameters
    num_envs: int = 1  # Number of environments
    total_timesteps: int = 100000  # Total number of timesteps
    gamma: float = 0.99  # discount factor
    lr_input_scaling: float = 0.01  # Learning rate for input scaling
    lr_weights: float = 0.01  # Learning rate for variational parameters
    lr_output_scaling: float = 0.01  # Learning rate for output scaling
    cuda: bool = False  # Whether to use CUDA
    num_qubits: int = 4  # Number of qubits
    num_layers: int = 2  # Number of layers in the quantum circuit
    device: str = "default.qubit"  # Quantum device
    diff_method: str = "backprop"  # Differentiation method
    save_model: bool = True # Save the model after the run

```

As you can see, the config is devided into 3 parts:

* **General parameters**: Here the name of your experiment as well as the logging path is defined. All metrics will be logged in a ```result.json``` file in the result folder which will have the time of the experiment execution as a prefix. You can also use [wandb](https://wandb.ai/site) for enhanced metric logging. 
* **Environment parameters**: This is in the simplest case just the string of the gym environment. For jumanji environments as well as for your custom environments, you can also specify additional parameters here (see #Tutorials for details).
* **Algorithms parameters**: All algorithms hyperparameters are specified here. For details on the parameters see [the algorithms section]()

Once you execute the file, it will create the subfolders and copy the config which is used for the experiment in the folder:

```py title="reinforce_quantum.py"
    config = vars(Config())
    
    # Based on the current time, create a unique name for the experiment
    config['trial_name'] = datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config['trial_name']
    config['path'] = os.path.join(Path(__file__).parent.parent, config['trial_path'], config['trial_name'])

    # Create the directory and save a copy of the config file so that the experiment can be replicated
    os.makedirs(os.path.dirname(config['path'] + '/'), exist_ok=True)
    config_path = os.path.join(config['path'], 'config.yml')
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    # Start the agent training 
    reinforce_quantum(config)   
```

After the execution, the experiment data is saved e.g. at: 

    ...
    configs
    examples
    logs/
        2025-03-04--14-59-32_reinforce_quantum          # The name of your experiment
            config.yaml                                 # Config which was used to run this experiment
            result.json                                 # Results of the experiment
    .gitignore
    ...


You can also set the ```wandb``` variable to ```True```:

```py title="reinforce_quantum.py" hl_lines="4 6"
@dataclass
class Config:
    # General parameters
    trial_name: str = 'reinforce_quantum_wandb'  # Name of the trial
    trial_path: str = 'logs'  # Path to save logs relative to the parent directory
    wandb: bool = True # Use wandb to log experiment data 
    project_name: str = "cleanqrl"  # If wandb is used, name of the wandb-project

    # Environment parameters
    env_id: str = "CartPole-v1" # Environment ID
```

You will need to login to your [wandb](https://wandb.ai/site) account before you can run:

```bash
wandb login # only required for the first time
python cleanrl/reinforce_quantum.py \
```

This will create an additional folder for the [wandb](https://wandb.ai/site) logging and you can inspect your experiment data also online:

    ...
    configs
    examples
    logs/
        2025-03-04--14-59-32_reinforce_quantum_wandb    # The name of your experiment
            wandb                                       # Subfolder of the wandb logging
            config.yaml                                 # Config which was used to run this experiment
            result.json                                 # Results of the experiment
    .gitignore
    ...

## Run experiments with config files

Additionally, all algorithms can be executed from an external script as well. There are two examples in the root directory ```main.py``` and ```main_batch.py```. You can specify all parameters in a YAML file instead (and also reuse the ```config.yaml```files which have been generated in previous runs). For examples, take a look at the ```configs/basic```folder. You will just need to specify the config path. 

```py title="main.py" hl_lines="1"
    config_path = "configs/basic/reinforce_quantum.yaml"

    # Load the config file
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Based on the current time, create a unique name for the experiment
    config["trial_name"] = (
        datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + "_" + config["trial_name"]
    )
    config["path"] = os.path.join(
        os.path.dirname(os.getcwd()), config["trial_path"], config["trial_name"]
    )

    # Create the directory and save a copy of the config file so
    # that the experiment can be replicated
    os.makedirs(os.path.dirname(config["path"] + "/"), exist_ok=True)
    shutil.copy(config_path, os.path.join(config["path"], "config.yaml"))

    # Start the agent training
    train_agent(config)
```

If you want to execute several config files sequentially, you can also you the ```main_batch.py``` file, where you can specify several configs in a list or execute all configs in a subdirectory.

```py title="main_batch.py" 
    # Specify the path to the config file
    # Get all config files in the configs folder
    # config_files = [f for f in os.listdir('configs/basic') if f.endswith('.yaml')]
    config_paths = [
        "configs/basic/dqn_classical.yaml",
        "configs/basic/reinforce_classical.yaml",
        "configs/basic/reinforce_classical_continuous_action.yaml",
        "configs/basic/ppo_classical.yaml",
        "configs/basic/ppo_classical_continuous_action.yaml",
    ]
```


## Experiment logging
By default, all metrics are logged to a ```result.json``` file on the experiment folder. Plots are generated by default for these runs as well for some of the metrics. For details, take a look at ```cleanqrl_utils/plotting.py```. 

When using ```wandb```, all data is additionally logged to your [wandb account](https://wandb.auth0.com/login?state=hKFo2SA0ZlFTcXRxUHpwbHRwc0pjamVoY2ZMUnJKc05hY2dpLaFupWxvZ2luo3RpZNkgTUQxN3NPVmVVN1ptT0ZCaURoajlONm1aT3BUdFd4Vi2jY2lk2SBWU001N1VDd1Q5d2JHU3hLdEVER1FISUtBQkhwcHpJdw&client=VSM57UCwT9wbGSxKtEDGQHIKABHppzIw&protocol=oauth2&nonce=Rn5hS353NjNrS1FUNEFqWA%3D%3D&redirect_uri=https%3A%2F%2Fapi.wandb.ai%2Foidc%2Fcallback&response_mode=form_post&response_type=id_token&scope=openid%20profile%20email&signup=true) .

