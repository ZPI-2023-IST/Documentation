Reinforcement Learning Framework Documentation
==============================================

=================
Description
=================

Reinforcement Learning Framework is used to train various models on various games.
The framework is built in such a way that minimal user input is required.

====================================================================
Structure of the Reinforcement Learning Framework
====================================================================

Framework consists of the following parts:

1. algorithms - module containing reinforcement learning algorithms

    a. modules - module containg neural networks implementation

        1. SimpleNet - class implementing a simple neural network

    b. Algorithm - abstract class containing methods that every algorithm must override
    c. AlgorithmManager - class managing all the implemented algorithms
    d. Config - class containing configuration for algorithms
    e. learning_algorithms/simple_algorithms - examples of algorithm implementation

2. api - module containing application runner and shared endpoints

    a. endpoints - implementation of shared endpoints
    b. main - method starting the Reinforcement Learning Framework
    c. Runner - class responsible for running the Reinforcement Learning Framework

3. logger - module containg logger functionalities

    a. Logger - class containing logger functionalities

--------------------------------------
Algorithms Module
--------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Module containing implementation of neural networks that could be later used to build reinforcement learning models. 
Neural networks do not need to adhere to any rules.
User can implement their own neural networks when needed

""""""""""""""""""""""""""""""""""""""
SimpleNet
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

""""""""""""""""""""""""""""""""""""""
SimpleNet - Methods
""""""""""""""""""""""""""""""""""""""

**__init__**

Initiate SimpleNet::

    def __init__(self, layers: list[int]):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.activation = nn.ReLU()

| Input: list of the number of neurons in each hidden layer (length of the list is the number of hidden layers)
| Output: None

**forward**

Forward propagate ::

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x

| Input: input vector
| Output: output vector

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The structure of Algorithm is the following::

    from abc import ABC, abstractmethod
    from collections import namedtuple

    from rl.algorithms.Config import Config

    Parameter = namedtuple(
        "Parameter", ("type", "default", "min", "max", "help", "modifiable")
    )


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

**Parameter**

 Tuple subclass for defining model parameters::

    Parameter = namedtuple(
        "Parameter", ("type", "default", "min", "max", "help", "modifiable")
    )

| type - type of the parameter (types defined in ParameterType in Config)
| default - default value of the parameter. Set to None if parameter doesn't have a default value
| min - minimal value of the parameter. Set to None if parameter doesn't have minimal value
| max - maximal value of the parameter. Set to None if parameter doesn't have maximal value
| help - description of the parameter
| modifiable - is parameter modifiable after training has started?

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

Configure model using given configuration ::

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
