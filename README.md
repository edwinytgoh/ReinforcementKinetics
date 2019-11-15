# CombustoRL

A reinforcement learning approach to designing high-temperature, low-NOx gas turbine combustors for next generation power plants.

## File Structure and Usage

The main environment file is found in envs/**sim_env.py**. This file defines the `SimEnv` class, which extends Open AI Gym's base [Env](https://github.com/openai/gym/blob/master/gym/core.py) class.

In addition, there are Jupyter notebooks in the root directory that import from sim_env.py:
    1. SimpleAgent.ipynb — train a simple PPO2 agent using a dummy vectorized environment and save the agent in the "Trained Models" folder.
    2. TestTrainedModel.ipynb — load parameters from the trained PPO2 agent and use it to "replay" the simulation. 
    3. MultiprocessingAgent.ipynb — train a PPO2 agent on a vectorized environment using multiple MPI processes. Still needs additional debugging. 
    4. TestSimEnv.ipynb — simple sanity checks to ensure SimEnv is instantiatble.

## Requirements

This repository uses libraries mostly found in the Anaconda repository. However, a few deep learning libraries are only available on PyPI. Users are referred to an excellent blog post describing best practices when using pip in a conda environment: <https://www.anaconda.com/using-pip-in-a-conda-environment/>

A yml file containing the conda environments used by the authors will be included soon.

### Key packages to install

1. [Cantera](https://cantera.org) — an open-source suite of tools for problems involving chemical kinetics, thermodynamics, and transport processes.
    - As of 11/08/2019, Cantera requires an earlier version of numpy, which *may get overriden by pip*. Please use –upgrade-strategy only-if-needed when installing packages through pip to make sure that this doesn't become an issue.
2. [Open AI Gym](https://gym.openai.com/) — a toolkit for developing and comparing reinforcement learning algorithms.
    - We use Gym to build a combustor simulation environment that will be controlled by an agent.
    - Note: According to the Gym readme, **Windows support is *experimental***.
3. [Stable Baselines](https://stable-baselines.readthedocs.io/en/master) — a set of improved implementations of Reinforcement Learning (RL) algorithms based on OpenAI Baselines.
   - Stable baselines in turn depends on Tensorflow between 1.8.0 and 1.14.0.
   - See <https://stable-baselines.readthedocs.io/en/master/guide/install.html> for further details on MPI dependencies or docker installation instructions.
