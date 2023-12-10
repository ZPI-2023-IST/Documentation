Build your own Translator module
===================================

===========================
How to build the module
===========================

To build the module you'll need to inherit the AbstractTranslator class :doc:`from the Translator module documentation <../modules/translator>` 
and implement all of the methods

======================================================
How to test if the module was implemented correctly
======================================================

| Verify that you can connect Game module to Translator module and call all Translator methods without Translator crashing
| Your Translator module should also follow these rules:

#. You should convert all moves to the list of indices. Model will return a given index that Translator should later convert to Game move
#. You should avoid passing any non-numeric values to the RL module
#. The only exception to the rule above is get_state function that can pass enum or string value

======================================================
Game module example (Freecell)
======================================================

""""""""""""""""""""""""""""""""""""""
Module structure
""""""""""""""""""""""""""""""""""""""

Among many files, our core functionality was split onto following files:
    * ``constants.py`` - contains all the constants used in the Freecell Translator
    * ``freecell_translator.py`` - contains the implementation of Freecell Translator
    * ``functions.py`` - contains all functions used in Freecell Translator

""""""""""""""""""""""""""""""""""""""
freecell_translator.py
""""""""""""""""""""""""""""""""""""""

This part of code is only meant to be a presentation on how the translator could look. 
For the full implementation look at `our repository <https://github.com/ZPI-2023-IST/Translator>`__.
Let's take a look at the ``FreecellTranslator`` class::

    from ..abstract_translator.abstract_translator import AbstractTranslator
    from .functions import *

    class FreecellTranslator(AbstractTranslator):
        def __init__(self, game=None):
            super().__init__(game)
            # ML vectors mapped to given index
            self.all_moves = self._get_all_moves_dict()
            self.all_moves_rev = {v:k for k,v in self.all_moves.items()}
            self.config_model = {
                "n_observations": np.prod(SIZE_BOARD) + np.prod(SIZE_FREE_CELL) + np.prod(SIZE_HEAP),
                "n_actions": len(self.all_moves)
            }

            # Store destination card for future reward calculation
            self.dst_card = None

        def make_move(self, move):
            ml_no_cards, ml_src, ml_dst = self.all_moves_rev[move]
            board, free_cells, _ = self.game.get_board()

            src_card = get_source_card(board, free_cells, ml_no_cards, ml_src)
            dst_card = get_dest_card(board, ml_dst)

            # Store destination card for future reward calculation
            self.dst_card = dst_card

            self.game.make_move((src_card, dst_card))

        # Returns list of index of all moves
        def get_moves(self):
            board, free_cells, _ = self.game.get_board()
            moves = self.game.get_moves()

            move_vectors = []
            for move in moves:
                src_card, dst_card = move
                cards_moved_vector, src_vector = get_source_card_vector(board, free_cells, src_card)
                dst_vector = get_dest_card_vector(board, dst_card)

                move_id = self.all_moves[(cards_moved_vector, src_vector, dst_vector)]
                move_vectors.append(move_id)

            return move_vectors

        # Our ml model takes one dimensional inputs
        def get_board(self):
            board, free_cells, heap = self.game.get_board()

            board = convert_board_to_ar_ohe(board)
            free_cells = convert_fc_to_ar_ohe(free_cells)
            heap = convert_heap_to_ar_ohe(heap)

            return np.concatenate((board, free_cells, heap)).tolist()

        def get_state(self):
            return self.game.get_state()
        
        def start_game(self):
            self.game.start_game()

        def get_reward(self):
            state = self.game.get_state()
            if state.value == State.WON.value:
                return 5
            elif state.value == State.LOST.value:
                return -5
            else:
                if self.dst_card == CARD_LOCATIONS.HEAP.value:
                    return 1
                else:
                    return 0   
        
        def get_config_model(self):
            return self.config_model
        
        def _get_all_moves_dict(self):
            result_dict = {}
            n_move = 0

            # Perform all one move cards 
            for src, src_v in CARDS_SOURCE.items():
                for dst, dst_v in CARDS_DEST.items():
                    if not self._is_the_same_col(src_v, dst_v):
                        no_cards = REV_NUMBER_OF_CARDS[1]
                        result_dict[(no_cards, src, dst)] = n_move
                        n_move += 1

            return result_dict 
        
        def _is_the_same_col(self, src, dst):
            if src == dst:
                return True
            
            if src[0] == CARD_LOCATIONS.FREE_CELL.value and dst[0] == CARD_LOCATIONS.FREE_CELL.value:
                return True
            
            return False

======================================================
Game module example (2048)
======================================================

""""""""""""""""""""""""""""""""""""""
Module structure
""""""""""""""""""""""""""""""""""""""

Among many files, our core functionality was split onto following files:
    * ``constants.py`` - contains all the constants used in the 2048 Translator
    * ``translator2048.py`` - contains the implementation of 2048 Translator=

""""""""""""""""""""""""""""""""""""""
translator2048.py
""""""""""""""""""""""""""""""""""""""

This part of code is only meant to be a presentation on how the translator could look. 
For the full implementation look at `our repository <https://github.com/ZPI-2023-IST/Translator_2048>`_.
Let's take a look at the ``Translator2048`` class::

    import math

    from .constants import *
    from ..abstract_translator.AbstractTranslator import AbstractTranslator

    class Translator2048(AbstractTranslator):

        def __init__(self, game=None):
            super().__init__(game)
            self.move_indexes = list(MOVES)

        def make_move(self, move_index):
            move_vector = self.move_indexes[move_index].value[1]
            matching_move = next(move for move in MOVES if move.value[1] == move_vector)
            move = matching_move.value[0]
            self.game.make_move((move,))
            return True

        def get_moves(self):
            all_moves = self.game.get_moves()
            moves_indexes = [self.move_indexes.index(get_enum_member(move)) for move in all_moves]
            return moves_indexes

        def get_board(self):
            board = self.game.get_board()
            board_one_hot_values = [FIELDS_VALUES[field.value] for row in board for field in row]
            return board_one_hot_values

        def get_state(self):
            return self.game.get_state()

        def start_game(self):
            self.game.start_game()

        def get_reward(self):
            state = self.game.get_state()
            if state.value == State.WON.value:
                return 10
            elif state.value == State.LOST.value:
                return -10
            else:
                # Modify merge_reward and empty_penalty to handle None values
                merge_reward = sum([tile.value for row in self.game.get_board() for tile in row if tile.value is not None])
                empty_penalty = -0.1 * len(
                    [tile.value for row in self.game.get_board() for tile in row if tile.value is None])

                monotonic_reward = self.__calculate_monotonic_reward()  # Reward for board monotonicity
                smoothness_reward = self.__calculate_smoothness_reward()  # Reward for smoothness

                total_reward = merge_reward + empty_penalty + monotonic_reward + smoothness_reward
                normalized_reward = math.log(total_reward + 1) / 2  # Logarithmic normalization
                scaled_reward = min(10, max(0, normalized_reward))  # Scale to be between 0 and 10

                return scaled_reward

        def get_config_model(self):
            pass

        def __calculate_smoothness_reward(self):
            smoothness_reward = 0
            board = self.game.get_board()
            for row in board:
                for i in range(1, len(row)):
                    if row[i].value is not None and row[i - 1].value is not None:
                        smoothness_reward -= abs(row[i].value - row[i - 1].value)

            for col in zip(*board):
                for i in range(1, len(col)):
                    if col[i].value is not None and col[i - 1].value is not None:
                        smoothness_reward -= abs(col[i].value - col[i - 1].value)

            return smoothness_reward

        def __calculate_monotonic_reward(self):
            monotonic_reward = 0
            board = self.game.get_board()

            for row in board:
                monotonic_reward += sum([abs(row[i].value or 0 - row[i - 1].value) for i in range(1, len(row)) if
                                        None not in (row[i].value, row[i - 1].value)])

            for col in zip(*board):
                monotonic_reward += sum([abs((col[i].value or 0) - (col[i - 1].value or 0)) for i in range(1, len(col)) if
                                        None not in (col[i].value, col[i - 1].value)])

           return monotonic_reward
