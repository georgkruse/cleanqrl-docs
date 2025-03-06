# Hyperparameter Tuning

## Ray Tune 

Generally, in (Q)RL, hyperparameter tuning is essential. Therefore, we offer easy hyperparamter tunings with [ray tune](). To use it, you can take a look at the config files located in the ```configs/tune``` folder and the ```tune.py```.

```py title="tune.py"
    # Generate the parameter space for the experiment from the config file
    config = add_hyperparameters(config)

    .... 

    # Instead of running a single agent as before, we will use ray.tune to run multiple agents
    # in parallel. We will use the same train_agent function as before.
    ray.init(
        local_mode=config["ray_local_mode"],
        num_cpus=config["num_cpus"],
        num_gpus=config["num_gpus"],
        _temp_dir=os.path.join(os.path.dirname(os.getcwd()), "t"),
        include_dashboard=False,
    )

    # We need an addtional function to create subfolders for each hyperparameter configuration
    def trial_name_creator(trial):
        return trial.__str__() + "_" + trial.experiment_tag

    # We will use the tune.Tuner class to run multiple agents in parallel
    trainable = tune.with_resources(
        train_agent,
        resources={"cpu": config["cpus_per_worker"], "gpu": config["gpus_per_worker"]},
    )
    tuner = tune.Tuner(
        trainable,
        param_space=config,
        run_config=tune.RunConfig(storage_path=config["path"]),
        tune_config=tune.TuneConfig(
            num_samples=config["num_samples"],
            trial_dirname_creator=trial_name_creator,
        ),
    )

    # The fit function will start the hyperparameter search
    tiral = tuner.fit()
```

We use the **tune.Tuner** to perform the hyperparameter search. But before we take a look at the hyperparameters, lets see first how to define the resources: In the config files, we have an additional block:

```yaml
# ray tune parameters
ray_local_mode:         False
num_cpus:               24
num_gpus:               0
num_samples:  3
cpus_per_worker:        1
gpus_per_worker:        0
```
The first parameter defines if we want to use the so called ```local_mode``` which enforces sequential execution instead of parallel execution for debugging purposes. Hence, this should always be False if you want to run the actual training. Then you need to specify the amount of cpus and gpus you want to make available to the **tune.Tuner**. This depends on the machine you are using. Next, you need to define the number of samples you want to run for each hyperparameter configuration. Generally, you do not want the number to be too small, because especially QRL experiments can have large variances in performance. But also you don't want this number to be too big, because this will cause very long runtimes. Lastly, you need to define how many resources each worker, that is each sample, gets to run. E.g. If you specify ```num_cpus=10``` and ```cpus_per_worker=1```, then 10 runs will be run in parallel.

## Add hyperparameters

Adding hyperparameters to sample is straight forward. For that, you can look at the following example. Instead of specifying e.g. 


```yaml 
lr_weights: 0.001 
```


 You can instead specify it as:

```yaml
lr: 
    - grid_search           # Do a grid search with number of seeds == num_samples
    - float                 # Define the type [float, int, str, list]
    - [0.01, 0.001, 0.0001] # List of parameters to select
``` 

If ```num_samples``` is set to 5 and ```num_cpus``` to 15, then each hyperparameter configuration will be run with 5 seeds in parallel (watch your RAM for larger qubit numbers).

You can now also edit other variables:

```yaml
batch_size: 
    - grid_search           # Do a grid search with number of seeds == num_samples
    - int                   # Define the type [float, int, str, list]
    - [16, 32]              # List of parameters to select
```

This will now start for each of the ```batch_size``` a hyperparameter run with all the specified learning rates for ```num_samples```, so a total of 30 trials. If ```num_cpus``` is set to 10, then it will sequentially execute the 30 trials with 10 trials in parallel.

You can even do a search over the environments as:

```yaml
env_id: 
    - grid_search           # Do a grid search with number of seeds == num_samples
    - str                   # Define the type [float, int, str, list]
    - ['CartPole-v1',       # List of parameters to select
       'Acrobot-v1']              
```

Finally, just like for the ```main.py```, there exists also a ```tune_batch.py``` file, where you can sequentially perform hyperparameter runs.