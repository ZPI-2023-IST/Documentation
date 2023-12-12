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

.. _algorithm:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Algorithm class delivers functions that every algorithm implemented in the framework must provide
| NOTE - every algorithm implemented needs to be registered by AlgorithmManager!

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

| Input: logger used to log features
| Output: None

**forward**

Make one iteration of the model ::

    @abstractmethod
    def forward(self, state: list, actions: list, reward: float) -> int:

| Input: current state of the game, list of possible actions, reward for previous action
| Output: the index of move chosen by the algorithm

**get_configurable_parameters**

Get all model's parameters ::

    @classmethod
    @abstractmethod
    def get_configurable_parameters(cls) -> dict:

| Input: None
| Output: List of all model's parameters

**get_model**

Get model data ::

    @abstractmethod
    def get_model(self) -> object:


| Input: None
| Output: TO DO

**set_params**

Load model parameters ::

    @abstractmethod
    def set_params(self, params) -> None:

| Input: model parameters
| Output: None

**config_model**

Create model configuration using Config class ::

    def config_model(self, config: dict) -> None:

| Input: dictionary containing documentation
| Output: None

**restart**

Restart defined model parameters ::

    def restart(self) -> None:

| Input: None
| Output: None

**update_config**

Update model configuration ::

    def update_config(self, config: dict) -> None:

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
AlgorithmManager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AlgorithmManager class handles all implemented algorithms. 
It registers all the implemented algorithms thus providing access to them to users. 
It also handles all operations related to setting up algorithms.

""""""""""""""""""""""""""""""""""""""
Methods
""""""""""""""""""""""""""""""""""""""

**__init__**

Initiate algorithm manager ::

    def __init__(self) -> None:

| Input: None
| Output: None

**mount**

Mount algorithm manager ::

    def mount(self, logger) -> None:

| Input: logger used to log messages
| Output: None

**set_default_algorithm**

Set default algorithm defined in the algorithm manager and configure it using default parameters ::

    def set_default_algorithm(self) -> None:

| Input: None
| Output: None

**set_algorithm**

Set algorithm and configure it using default parameters. Log changes ::

    def set_algorithm(self, algorithm_name: str, *args, **kwargs) -> None:

| Input: name of the algorithm
| Output: None

**configure_algorithm**

Create new configuration for currently used algorithm. Log changes ::

    def configure_algorithm(self, config: dict) -> None:

| Input: new configuration for currently used algorithm
| Output: None

**update_config**

Update configuration of the currently used algorithm. Log changes  ::

    def update_config(self, config: dict) -> None:

| Input: new configuration for currently used algorithm
| Output: None

**register_algorithm**

Register the algorithm. Every implemented algorithm needs to be registered with the usage of register_algorithm decorator ::

    def register_algorithm(self, name: str):

| Input: name of the algorithm
| Output: None

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Config class stores configuration of the algorithms. Every algorithm uses config to access theirs parameters

""""""""""""""""""""""""""""""""""""""
Methods
""""""""""""""""""""""""""""""""""""""

**from_dict**

Create new configuration from dictionary ::

    @staticmethod
    def from_dict(config: dict) -> None:

| Input: dictionary with configuration
| Output: None

**as_dict**

Get configuration as dictionary ::

    def as_dict(self) -> dict:

| Input: None
| Output: dictionary with configuration

**update**

Update configuration ::

    @staticmethod
    def update(self, config: dict) -> None:

| Input: dictionary with configuration
| Output: None

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
States
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enum defining if the model is currently in train or test mode ::

    from enum import Enum

    class States(Enum):
        TRAIN = "train"
        TEST = "test"

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ParameterType
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enum defining possible model parameter types. Currently it allows for parameters to have 4 types ::

    from enum import Enum, auto

    class ParameterType(Enum):
        INT = auto()
        FLOAT = auto()
        BOOL = auto()
        STRING = auto()

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

--------------------------------------
API Module Structure
--------------------------------------

Algorithms Module consists of the following parts:

#. main
#. endpoints
#. Runner class
#. GameResults class
#. GameState enum

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
main
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

main file is used to start the reinforcement learning framework server. 
You need to start the server if you want to use the framework

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Endpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| We share multiple endpoints that allow to use reinforcement learning framework methods
| To use them you need to use HTTP methods: GET, POST, PUT

**/logs**

| GET - get logs from the server

**/run**

| GET - get current runner state
| POST - start/stop runner

**/model**

| GET - return model in zip format
| PUT - import model from the file

**/config**

| GET - get current model configuration
| PUT - update model configuration
| POST - create new model configuration

**/config-params**

| GET - get all model parameters (you can choose if you want modifiable parameters or all parameters)

**/game-history**

| GET - return game history

**/stats**

| GET - return statistics about training/testing process

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Runner
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Runner class is an intermediary between algorithms module, API and Frontend

""""""""""""""""""""""""""""""""""""""
Methods
""""""""""""""""""""""""""""""""""""""

**__init__**

Initiate runner ::

    def __init__(self, logger: Logger, algorithm_manager: AlgorithmManager, max_game_len=100, config="config.json") -> None:

| Input: logger, algorithm manager, maximal amount of moves that the model can make in one iteration of the game, configuration file for connecting with API
| Output: None

**_mount_socketio**

Initiate connection to API ::

    def _mount_socketio(self) -> None:

| Input: None
| Output: None

**time**

Get the running time of runner ::

    @property
    def time(self) -> float:

| Input: None
| Output: None

**run**

Run the runner until end condition is met, user stops the process or the connection gets lost ::

    def run(self) -> None:

| Input: None
| Output: None

**start**

Start the runner ::

    def start(self) -> None:

| Input: None
| Output: None

**stop**

Stop the runner ::

    def stop(self) -> None:

| Input: None
| Output: None

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
GameResults
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GameResults class stores model training/testing results

""""""""""""""""""""""""""""""""""""""
Methods
""""""""""""""""""""""""""""""""""""""

**__init__**

Initiate GameResults class. Set all statistics to 0 ::

    def __init__(self) -> None:

| Input: None
| Output: None

**store_game_results**

Store game results from one iteration ::

    def store_game_results(self, reward, game_status, is_end_game):

| Input: reward from the current game, game status (won, lost, ongoing), is game finished
| Output: None

**__str__**

Get train/test statistics as string ::

    def __str__(self) -> str:

| Input: None
| Output: train/test statistics as string

**get_results**

Get train/test statistics as dictionary ::

    def get_results(self):

| Input: None
| Output: train/test statistics as dictionary

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
GameState
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enum defining the current state of the game ::

    from enum import Enum, auto

    class GameStates(Enum):
        ONGOING = auto()
        WON = auto()
        LOST = auto()

--------------------------------------
Logger Module Structure
--------------------------------------

Logger Module consists of the following parts:

#. Logger class
#. LogLevel enum
#. LogType enum

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Logger
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Logger class is used to log operations made within the framework

""""""""""""""""""""""""""""""""""""""
Methods
""""""""""""""""""""""""""""""""""""""

**__init__**

Initiate logger ::

    def __init__(self) -> None:

| Input: None
| Output: None

**log**

Print log message and save it in messages ::

    def log(self, message: str, log_level: LogLevel, log_type: LogType) -> None:

| Input: message, log level, log type
| Output: None

**info**

Log message of into type  ::

    def info(self, message: str, log_type: LogType) -> None:

| Input: message, log type
| Output: None

**get_messages**

Get all logged messages ::

    def get_messages(self, filter: str = None) -> list:

| Input: condition on which messages will be filtered. If None than no filter is applied
| Output: list of all the messages

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
LogLevel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enum defining level of the message ::

    from enum import Enum, auto

    class LogLevel(Enum):
        DEBUG = auto()
        INFO = auto()
        WARNING = auto()
        ERROR = auto()
        FATAL = auto()

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
LogType
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enum defining the type of the message ::

    from enum import Enum, auto

    class LogType(Enum):
        CONFIG = auto()
        TRAIN = auto()
        TEST = auto()
