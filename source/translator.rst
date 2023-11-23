Translator Documentation
===============================

=================
Description
=================

Translator is a intermediary between game and reinforcement learning models. You can also say
that translator is a way to make game compatible with reinforcement learning models

=================
Functionality
=================

#. Convert output of the reinforcement learning model to the representation that could be used by the game
#. Convert game moves to representation that could be used by the reinforcement learning model
#. Convert game board to representation that could be used by the reinforcement learning model
#. Convert game state to representation that could be used by the reinforcement learning model
#. Start game
#. Calculate reward for reinforcement learning models
#. Get translator parameters that could be useful when setting up reinforcement learning models

==================================
Structure of the Translator
==================================
| Every translator needs to inherit the AbstractTranslator class and implement all the methods

The structure of AbstractTranslator is the following::

    from abc import ABC, abstractmethod

    class AbstractTranslator(ABC):
        def __init__(self, game=None):
            self.game = game

        @abstractmethod
        def make_move(self, move):
            pass

        @abstractmethod
        def get_moves(self):
            pass

        @abstractmethod
        def get_board(self):
            pass

        @abstractmethod
        def get_state(self):
            pass

        @abstractmethod
        def start_game(self):
            pass

        @abstractmethod
        def get_reward(self):
            pass

        @abstractmethod
        def get_config_model(self):
            pass

==================================
Methods
==================================

---------------------------------
__init__
---------------------------------

Initiate translator::

    def __init__(self, game=None):
        self.game = game

| Input: game that is processed by translator
| Output: None

---------------------------------
make_move
---------------------------------

Make move in the game::

    def make_move(self, move):
        pass

| Input: move made by the reinforcement learning model
| Output: None

---------------------------------
get_moves
---------------------------------

Get moves from the game::

    def get_moves(self):
        pass

| Input: None
| Output: move representation needed by reinforcement learning model

---------------------------------
get_board
---------------------------------

Get board from the game::

    def get_board(self):
        pass

| Input: None
| Output: board representation needed by reinforcement learning model

---------------------------------
get_state
---------------------------------

Get state from the game::

    def get_state(self):
        pass

| Input: None
| Output: state representation needed by reinforcement learning model

---------------------------------
start_game
---------------------------------

Start the game::

    def start_game(self):
        pass

| Input: None
| Output: None

---------------------------------
get_reward
---------------------------------

Calculate reward that will be later used by reinforcement learning models::

    def get_reward(self):
        pass

| Input: None
| Output: reward for the reinforcement learning models

---------------------------------
get_config_model
---------------------------------

Get parameters that could be helpful when setting up the reinforcement learning model (e.g. input size, output size)::

    def get_config_model(self):
        pass

| Input: None
| Output: useful parameters when setting up the reinforcement learning model
