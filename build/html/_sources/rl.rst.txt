Reinforcement Learning Framework Documentation
==============================================

=================
Description
=================

Reinforcement Learning Framework is used to train various models on various games.
The framework is built in such a way that minimal user input is required.

=================
Functionality
=================

#. Share endpoints that allows API and Frontend to connect 
#. Collect game data from API
#. Collect data about model training and testing and pass them to API and Frontend
#. Pass list of all models with their parameters to Frontend

====================================================================
Structure of the Reinforcement Learning Framework
====================================================================

Framework consists of the following parts:

#. algorithms - module containing reinforcement learning algorithms and neural networks implementation
#. api - module containing application runner and shared endpoints
#. logger - module containg logger functionalities

--------------------------------------
Algorithms Module Structure
--------------------------------------

Algorithms Module consists of the following parts:

#. Algorithm class
#. modules module containing neural networks implementations
#. AlgorithmManager class
#. Config class
#. States enum
#. ParameterType enum
#. Parameter tuple subclass

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Algorithm class delivers functions that every algorithm implemented in the framework must provide

The structure of Algorithm class is the following::

    from abc import ABC, abstractmethod
    from rl.algorithms.Config import Config

    class Algorithm(ABC):
        def __init__(self, logger) -> None:
            self.logger = logger
            self.config = None

        @abstractmethod
        def forward(self, state: list, actions: list, reward: float, game_status: str) -> int:
            pass

        @classmethod
        @abstractmethod
        def get_configurable_parameters(cls) -> dict:
            pass

        @abstractmethod
        def get_model(self) -> object:
            pass

        @abstractmethod
        def set_params(self, params) -> object:
            pass

        def config_model(self, config: dict) -> None:
            self.config = Config.from_dict(config)

        def restart(self) -> None:
            pass
        
        def update_config(self, config: dict) -> None:
            self.config.update(config)

""""""""""""""""""""""""""""""""""""""
Methods
""""""""""""""""""""""""""""""""""""""

**__init__**

Initiate algorithm ::

    def __init__(self, logger) -> None:
        self.logger = logger
        self.config = None

| Input: logger used to log features
| Output: None

**forward**

Make one iteration of the model ::

    @abstractmethod
    def forward(self, state: list, actions: list, reward: float) -> int:
        pass

| Input: current state of the game, list of possible actions, reward for previous action
| Output: the index of move chosen by the algorithm

**get_configurable_parameters**

Get all model's parameters ::

    @classmethod
    @abstractmethod
    def get_configurable_parameters(cls) -> dict:
        pass

| Input: None
| Output: List of all model's parameters

**get_model**

Get model data ::

    @abstractmethod
    def get_model(self) -> object:
        pass


| Input: None
| Output: TO DO

**set_params**

Load model parameters ::

    @abstractmethod
    def set_params(self, params) -> None:
        pass

| Input: model parameters
| Output: None

**config_model**

Configure model using Config class ::

    def config_model(self, config: dict) -> None:
        self.config = Config.from_dict(config)

| Input: dictionary containing documentation
| Output: None

**restart**

Restart defined model parameters ::

    def restart(self) -> None:
        pass

| Input: None
| Output: None

**update_config**

Update model configuration ::

    def update_config(self, config: dict) -> None:
        self.config.update(config)

| Input: dictionary containing model configuration
| Output: None

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Module containing implementation of neural networks that could be later used to build reinforcement learning models. 
| Neural networks do not need to adhere to any rules.
| User can implement their own neural networks when needed

""""""""""""""""""""""""""""""""""""""
SimpleNet - example of neural network
""""""""""""""""""""""""""""""""""""""

Class implementing a multilayer perceptron with ReLU as activation function

The structure of SimpleNet is the following::

    import torch
    import torch.nn as nn


    class SimpleNet(nn.Module):
        def __init__(self, layers: list[int]) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
            )
            self.activation = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
                x = self.activation(x)
            return x

| SimpleNet takes as an input the list of the number of nodes in the hidden layers
| Than during forward method it performs a simple forward propagation

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 Tuple subclass for defining model parameters::

    from collections import namedtuple

    Parameter = namedtuple(
        "Parameter", ("type", "default", "min", "max", "help", "modifiable")
    )

| type - type of the parameter (types defined in ParameterType in Config)
| default - default value of the parameter. Set to None if parameter doesn't have a default value
| min - minimal value of the parameter. Set to None if parameter doesn't have minimal value
| max - maximal value of the parameter. Set to None if parameter doesn't have maximal value
| help - description of the parameter
| modifiable - is parameter modifiable after training has started?
