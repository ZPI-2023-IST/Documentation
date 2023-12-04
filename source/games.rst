Game modules
================================

================================
Description
================================

The environment for conducting reinforcement learning experiments is called a game.
While it does not explicitly have to be a game, it should contain a certain set of functionalities mentioned later.
The game module is considered a place for training and testing RL models.

================================
Functionality
================================

#. Contains built-in environment logic (e.g. rules of the game)
#. Should provide functions and methods for interacting with the environment

================================
Game module structure
================================

""""""""""""""""""""""""""""""""
Prerequisites
""""""""""""""""""""""""""""""""

The game module should contain an abstract class called ``Game`` and an enum class ``State``.

Example of Game and State in Python::

    from enum import Enum
    from abc import ABC, abstractmethod

    class State(Enum):
        ONGOING = 0
        WON = 1
        LOST = 2

    class Game(ABC):
        @abstractmethod
        def get_moves(self) -> list:
            pass

        @abstractmethod
        def make_move(self, move: tuple) -> bool:
            pass

        @abstractmethod
        def get_state(self) -> State:
            pass

        @abstractmethod
        def get_board(self) -> list:
            pass

        @abstractmethod
        def start_game(self) -> None:
            pass
            
""""""""""""""""""""""""""""""""""""""
State enum class
""""""""""""""""""""""""""""""""""""""

State enum class provides a highly readable way of describing the state of the game.
Using such a class is highly recommended as it makes it easy to tell whether model needs to undertake another game and to check its outcome.

""""""""""""""""""""""""""""""""""""""
Game functions and methods
""""""""""""""""""""""""""""""""""""""

**game_moves**

| Input: None
| Output: list of possible moves

**make_move**

| Input: move described as a tuple
| Output: boolean value indicating whether the move was successful

**get_state**

| Input: None
| Output: one of ``State`` values

**get_board**

| Input: None
| Output: list representing the board

**start_game**

| Input: None
| Output: None