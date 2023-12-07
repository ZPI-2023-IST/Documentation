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

================================
Game implementation examples
================================

""""""""""""""""""""""""""""""""""""""
FreeCell
""""""""""""""""""""""""""""""""""""""

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Module structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Among many files, our core functionality was split onto following files:
    * ``Board.py`` - contains ``Board`` class; represents game state and provides functionalities regarding performing moves
    * ``Card.py`` - contains ``Card`` class; provides card representation and functionalities regarding comparison and move validity checks
    * ``Deck.py`` - contains ``Deck`` class; defines ``Deck`` as a list of ``Card`` objects and provides shuffling functionality
    * ``Freecell.py`` - contains ``FreeCell`` class; game entry point, equipped only with methods required by the ``Game`` abstract class
    * ``Game.py`` - contains ``Game`` abstract class and ``State`` enum class
    * ``Move.py`` - contains ``Move`` enum class

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Game notation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| As described in ``Game`` abstract class, move in ``get_moves`` should be passed as a (preferably string) tuple.
| How does this approach fare in case of FreeCell?
| We've decided to simplify the notation as much as possible, resulting in (card, destination) format, where destination is either:

* ``'F'`` for an empty free cell
* ``'S'`` for suit stack
* ``'{rank}{color}'`` for any other destination card
* ``'0'`` for empty columns
* examples:
    - ``('TH', 'F')`` means moving the Ten of Hearts to a freecell
    - ``('JS', 'S')`` means moving the Jack of Spades to a suit stack
    - ``('AD', '2C')`` means moving the Ace of Diamonds to the 2 of Clubs
    - ``('AD', '0')`` means moving the Ace of Diamonds to an empty column

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FreeCell.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Consider games listed here only as an example and a brief guide on how to implement the necesities provided by ``Game``.
The part of showcased code is the most vital part of FreeCell in regards to entire project.
The rest of the code is available in `our repository <https://github.com/ZPI-2023-IST/FreeCell>`_.
Let's take a look at the ``FreeCell`` class, the entry point of the game::

    from game.Game import Game, State
    from game.Board import Board
    from game.Deck import Deck
    from game.Move import Move
    from random import Random


    class FreeCell(Game):
        def __init__(self, seed: int = None):
            if seed is None:
                seed = Random().randint(0, 1000000)
            self._move_count = 0
            self.deck = Deck(seed)
            self.board = Board(self.deck.cards_shuffled())

        def increment_move_count(self):
            self._move_count += 1

        # Overridden functions from game class

        def get_moves(self) -> list:
            """Get all possible moves from the current board state.

            :return: A list of all possible moves from the current board state.
            """
            moves = list()

            # Moves onto empty columns
            if [] in self.board.columns:
                for card in self.board.free_cells + self.board.get_movable_cards():
                    if card:
                        moves.append((str(card), "0"))

            # Get cards from the top of columns
            suspected_moves = self.board.get_movable_cards()

            # Check if at least one of freecells is empty
            if None in self.board.free_cells:
                # Append moving every from the top of column to a freecell
                for card in suspected_moves:
                    moves.append((str(card), "F"))

            for card in self.board.free_cells:
                if card:
                    # Check for suit stack moves
                    if card.is_larger_and_same_suit(self.board.suit_stack[card.suit]):
                        moves.append((str(card), "S"))

                    # Check if any card from freecells can be moved onto a column
                    for card_destination in suspected_moves:
                        if card.is_smaller_and_different_color(card_destination):
                            moves.append((str(card), str(card_destination)))

            for card in suspected_moves:
                # Check if any card from columns can be moved onto a suit stack
                if card.is_larger_and_same_suit(self.board.suit_stack[card.suit]):
                    moves.append((str(card), "S"))

                # Check if any card from columns can be moved onto another column
                for card_destination in suspected_moves:
                    if card != card_destination and (
                        card.is_smaller_and_different_color(card_destination)
                    ):
                        moves.append((str(card), str(card_destination)))

            return moves

        def make_move(self, move: tuple) -> bool:
            if move not in self.get_moves():
                # return False
                raise ValueError("Invalid move, not in get_moves()")

            card = self.board.find_card_from_string(move[0])
            match move[1]:
                case Move.FREECELL.value:
                    move_completed = self.board.move_to_free_cell(card)
                case Move.SUIT_STACK.value:
                    move_completed = self.board.move_to_stack(card)
                case Move.EMPTY_COLUMN.value:
                    move_completed = self.board.move_to_free_column(card)
                case _:
                    move_completed = self.board.move_to_card(
                        card, self.board.find_card_from_string(move[1])
                    )
            if move_completed:
                self.increment_move_count()
            else:
                raise ValueError("Invalid move, problem with execution")
            return move_completed

        def get_state(self) -> State:
            """Get the current state of the game.

            :return: The current state of the game as State enum.
            """
            suit_stack = list(self.board.suit_stack.values())
            for card in suit_stack:
                if card is None or card.rank != 13:
                    return State.ONGOING if bool(self.get_moves()) else State.LOST
            return State.WON

        def get_board(self) -> list:
            """Get the current board state.

            :return: The current board state as a list of 10 lists:
                * The first 8 lists are the columns.
                * The next 4 element long list is the list of free cells.
                * The last 4 element long list is
                the list of the top cards on each suit stack.
            """
            return (
                self.board.columns,
                self.board.free_cells,
                list(self.board.suit_stack.values()),
            )

        def start_game(self) -> None:
            self.__init__()


""""""""""""""""""""""""""""""""""""""
2048
""""""""""""""""""""""""""""""""""""""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Module structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Among many files, our core functionality was split onto following files:
    * ``board.py`` - contains ``Board`` class; represents game state and provides functionalities regarding performing moves and their validity checks
    * ``game.py`` - contains ``Game`` abstract class and ``State`` enum class
    * ``game2048.py`` - contains ``Game2048`` class and ``Direction`` enum class; game entry point, equipped only with methods required by the ``Game`` abstract class and an enum representing move directions
    * ``node.py`` - contains a brief ``Node`` class; representation of a game tile 

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Game notation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| As described in ``Game`` abstract class, move in ``get_moves`` should be passed as a (preferably string) tuple.
| How does this approach fare in case of 2048?
| Considering the overall simplicity of the game the notation is (direction, ) where direction is:

* ``'w'`` for a move up
* ``'s'`` for a move down
* ``'a'`` for a move left
* ``'d'`` for a move right

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
game2048.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Consider games listed here only as an example and a brief guide on how to implement the necesities provided by ``Game``.
The part of showcased code is the most vital part of 2048 in regards to entire project.
The rest of the code is available in `our repository <https://github.com/ZPI-2023-IST/2048>`_.
Let's take a look at the ``Game2048`` class, the entry point of the game::

    from code2048.game import Game, State
    from code2048.board import Board


    class Game2048(Game):
        def __init__(self, board: Board = None, rows: int = 4, cols: int = 4) -> None:
            self.board = board if board else Board(rows, cols)

        def get_moves(self) -> list:
            """
            Provides possible moves as a list of w/s/a/d characters meaning up/down/left/right respectively
            """
            return [key.value for key in self.board.possible_moves.keys()]

        def make_move(self, move: tuple) -> bool:
            """
            Returns True if move succeeded, False otherwise.

            Requires move in form of one element tuple, containing character mentioned above.

            Example: make_move('w',) will perform an upwards move.
            """

            if move[0] in self.get_moves():
                self.board.make_move(move[0])
                return True
            return False

        def get_state(self) -> State:
            """
            Returns game state enum:  State.{ONGOING / WON / LOST}.
            """
            return self.board.game_status()

        def get_board(self) -> list:
            """
            Returns current board state as a list of lists (rows).
            """
            return self.board.board

        def start_game(self) -> None:
            """
            Overwrites current object, invoking constructor with default values and resetting every variable.
            """
            self.board = Board()
