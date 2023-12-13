Tests
================================

This file contains the description of all the tests written in the framework

================================
Unit Tests
================================

--------------------------------------
Game Module
--------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Freecell
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All the tests are located in the `FreeCell repository tests folder <https://github.com/ZPI-2023-IST/FreeCell/tree/master/game/tests>`_.

**How to run tests**

To run tests you'll need to:

#. :ref:`Setup the game module <api_setup>`
#. cd game
#. python -m unittest 

If you did not modify anything all tests should pass

**Tests description**

Our tests are spread across 4 files:

#. test_Board.py - tests if the Board class works correctly
#. test_Card.py - tests if the Card class works correctly
#. test_Deck.py - tests if the Deck class works correctly
#. test_Freecell.py - tests if the FreeCell class works correctly

* **test_Board.py**

    - **test_empty_cells_all_board_empty**

        -  Goal - test if the empty_cells method works correctly when all cells are empty
        -  Precondition - None
        -  Excepted results - the method should return 4 (number of freecells) + 8 (number of columns) = 12

    - **test_empty_cells_and_move_to_free_cell_real_moves**

        -  Goal - test if the empty_cells method works correctly when there are real moves
        -  Precondition - None
        -  Excepted results - the method should return various numbers based of tested scenario

    - **test_make_deck**

        -  Goal - tests integrity of cards
        -  Precondition - None
        -  Excepted results - mocked cards should be arranged to predefined expected outcome

    - **test_move_to_stack_only_aces**
    
        -  Goal - test if the move_to_stack method works correctly when there are only aces
        -  Precondition - None
        -  Excepted results - the test should tell if moving cards to respective suit stacks was successful

    - **test_move_to_stack_aces_then_2_and_3**

        -  Goal - extension of previous test, works with aces, 2s and 3s
        -  Precondition - None
        -  Excepted results - the test should tell if moving cards to respective suit stacks was successful

    - **test_move_to_card_single_card_move**
    
        -  Goal - test if the move_to_card method works correctly when there is only one card to move
        -  Precondition - None
        -  Excepted results - the test should tell if moving cards onto other cards was successful

    - **test_move_to_free_column**
    
        -  Goal - test if the move_to_free_column method works correctly
        -  Precondition - None
        -  Excepted results - the test should tell if moving cards to free columns was successful

    - **test_move_card_from_free_cell_to_card**
        
        -  Goal - test if move_to_free_cell and move_to_card methods works correctly
        -  Precondition - None
        -  Excepted results - the test should tell if moving cards from free cells to columns was successful

    - **test_move_card_from_free_cell_to_column**
        
        -  Goal - test if move_to_free_cell and move_to_free_column methods work correctly
        -  Precondition - None
        -  Excepted results - the test should tell if moving cards from free cells to free columns was successful

* **test_Card.py**

    - **test_is_smaller_and_different_color**

        -  Goal - test if the is_smaller_and_different_color method works correctly when cards are different colors
        -  Precondition - None
        -  Excepted results - the test should pass if the card is smaller and different color

    - **test_is_larger_and_same_suit**
    
        -  Goal - test if the is_larger_and_same_suit method works correctly when cards are the same suit
        -  Precondition - None
        -  Excepted results - the test should pass if the card is larger and same suit

    - **test_eq**

        -  Goal - test if the eq method works correctly
        -  Precondition - None
        -  Excepted results - the test should pass if the cards are equal

    - **test_repr**

        -  Goal - test if the repr method works correctly
        -  Precondition - None
        -  Excepted results - the test should pass if repr is equal to desired card representation

    - **test_str**
    
        -  Goal - test if the str method works correctly
        -  Precondition - None
        -  Excepted results - the test should pass if str is equal to desired card representation

* **test_Deck.py**

    - **test_initialization_default**

        -  Goal - test if the deck is initialized correctly with no seed passed
        -  Precondition - None
        -  Excepted results - the test should pass if the deck's seed is equal to 1

    - **test_initialization_custom**

        -  Goal - test if the deck is initialized correctly with custom seed
        -  Precondition - None
        -  Excepted results - the test should pass if the deck's seed is equal to passed seed

    - **test_repr_str**

        -  Goal - test if the repr and str outcomes are equal
        -  Precondition - None
        -  Excepted results - the test should pass if the repr and str outcomes are equal

    - **test_cards_shuffled**
    
        -  Goal - test if the cards are shuffled
        -  Precondition - None
        -  Excepted results - the test should pass if the cards are shuffled (i.e. not equal to the previous order)

    - **test_custom_seed**

        -  Goal - test if the deck is initialized correctly with custom seed
        -  Precondition - None
        -  Excepted results - the test should pass if the deck's seed is equal to passed seed and if deck shuffles are consistent within the seed

* **test_Freecell.py**

    - **test_scenario_in_progress**

        -  Goal - overall test of the game state for a given scenario
        -  Precondition - None
        -  Excepted results - the test should pass if the game state is *ONGOING* and the action sequence is successful

    - **test_scenario_20_moves**
    
        -  Goal - overall test of the game performance after making 20 moves
        -  Precondition - None
        -  Excepted results - the test should pass if the move sequence is successful

    - **test_scenario_empty_board**

        -  Goal - assert variables for an empty board scenario
        -  Precondition - None
        -  Excepted results - the test should pass if variables have desired values

    - **test_scenario_no_moves**
    
        -  Goal - assert variables for a scenario with no moves
        -  Precondition - None
        -  Excepted results - the test should pass if variables have desired values and game is in *LOST* state

    - **test_scenario_free_column**

        -  Goal - overall test of performing a move to free column
        -  Precondition - None
        -  Excepted results - the test should pass if variables have desired values after performing the move

    - **test_scenario_stack_move**

        -  Goal - overall test of performing moves to suit stacks
        -  Precondition - None
        -  Excepted results - the test should pass if variables have desired values after performing moves

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2048
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All the tests are located in the `2048 repository tests folder <https://github.com/ZPI-2023-IST/2048/tree/master/code2048/tests>`_.

**How to run tests**

To run tests you'll need to:

#. :ref:`Setup the game module <api_setup>`
#. cd code2048
#. python -m unittest 

If you did not modify anything all tests should pass

**Tests description**

Our tests are spread across 2 files:

#. test_board.py - tests if the Board class works correctly
#. test_node.py - tests if the Node class works correctly

* **test_board.py**

    - **test_init**

        -  Goal - test if the constructor works correctly for various parameters
        -  Precondition - None
        -  Excepted results - several boards should be initialized correctly

    - **test_spawn**

        -  Goal - test if the spawn method works correctly
        -  Precondition - None
        -  Excepted results - the board should spawn a new tile in a random empty cell, given there is at least one

    - **test_possible_moves**

        -  Goal - test if the possible_moves are saved properly
        -  Precondition - None
        -  Excepted results - variable possible_moves should contain up to 4 possible moves, depending on scenario

    - **test_transpose**

        -  Goal - test if the transpose method works correctly
        -  Precondition - None
        -  Excepted results - the board should be transposed correctly

    - **test_move_right**

        -  Goal - test if the move_right method works correctly
        -  Precondition - None
        -  Excepted results - all tiles should slide as far right as possible

    - **test_move_left**

        -  Goal - test if the move_left method works correctly
        -  Precondition - None
        -  Excepted results - all tiles should slide as far left as possible

    - **test_move_up**

        -  Goal - test if the move_up method works correctly
        -  Precondition - None
        -  Excepted results - all tiles should slide as far up as possible

    - **test_move_down**    

        -  Goal - test if the move_down method works correctly
        -  Precondition - None
        -  Excepted results - all tiles should slide as far down as possible

* **test_node.py**

    - **test_create_node**

        -  Goal - test if the constructor works correctly for various values
        -  Precondition - None
        -  Excepted results - nodes should be initialized only for positive integes

    - **test_double**
    
        -  Goal - test if the double method works correctly
        -  Precondition - None
        -  Excepted results - the node should double its value

--------------------------------------
RL Module
--------------------------------------

All the tests are located in the `RL module repository tests folder <https://github.com/ZPI-2023-IST/RL/tree/master/rl/tests>`_.

**How to run tests**

To run tests you'll need to:

#. :ref:`Setup the RL module <rl_setup>`
#. cd rl
#. python -m unittest 

If you did not modify anything all tests should pass

**Tests description**

Our tests are spread across 5 files:

#. test_algorithm_manager.py - tests if the AlgorithmManager class works correctly
#. test_algorithms.py - tests if the Algorithm class works correctly
#. test_api.py - tests if endpoints work correctly
#. test_dqn.py - tests if the DQN class works correctly
#. test_logger.py - tests if the Logger class works properly

* **test_algorithm_manager.py**

    - **test_decorator**

        -  Goal - test if the new algorithm can be properly registered
        -  Precondition - the new algorithm wasn't registered before
        -  Excepted results - new algorithm should be registered by algorithm manager

    - **test_set_algorithm**

        -  Goal - test if the algorithm can be set as the current algorithm
        -  Precondition - the algorithm is not set as the current algorithm
        -  Excepted results - the algorithm should be set as the current algorithm

    - **test_default_algorithm**

        -  Goal - test if algorithm manager can set the default algorithm as the current algorithm
        -  Precondition - None
        -  Excepted results - the default algorithm should be set as the current algorithm

    - **test_configure_algorithm**

        -  Goal - test if algorithm manager can change the parameter values of the current algorithm
        -  Precondition - None
        -  Excepted results - the parameter values of the current algorithm should be changed

* **test_algorithms.py**

    - **test_config**

        -  Goal - test if config can be properly set for the algorithm
        -  Precondition - None
        -  Excepted results - config is set properly for the algorithm

    - **test_random**
        -  Goal - test if forward method in algorithm works correctly (check if random algorithm can pick an action)
        -  Precondition - None
        -  Excepted results - algorithm should always choose the action from the list of actions

    - **test_registered_algorithms**

        -  Goal - test if all algorithms registered inherit from the Algorithm class
        -  Precondition - there is at least one algorithm registered
        -  Excepted results - all algorithms registered should inherit from the Algorithm class

    - **test_configurable_params**

        -  Goal - test if you can get a dictionary of configurable parameters from the algorithm
        -  Precondition - None
        -  Excepted results - algorithm returned proper dictionary of configurable parameters

* **test_api.py**

    Before tests begin we set client that connects to RL module server

    - **test_config_endpoint**

        -  Goal - test if you can read and modify model configuration using /config endpoint
        -  Precondition - None
        -  Excepted results - model configuration should be properly read and modified

    - **test_algorithm_update**

        -  Goal - test if you can create new model with the given dictionary of parameters using /config endpoint
        -  Precondition - None
        -  Excepted results - new model should be created and set with the given dictionary of parameters

    - **test_configurable_params**

        -  Goal - test if you can get a dictionary of configurable parameters using /config-params endpoint
        -  Precondition - None
        -  Excepted results - endpoint should return a proper dictionary of configurable parameters

    - **test_logs_endpoint**

        -  Goal - test if you can get a dictionary of logs using /logs endpoint
        -  Precondition - None
        -  Excepted results - endpoint should return a proper dictionary of logs

    - **test_model_endpoint**

        -  Goal - test if you can get current model parameters using /model endpoint
        -  Precondition - None
        -  Excepted results - endpoint should return current model parameters

* **test_dqn.py**

    Before tests begin we setup DQN algorithm

    - **test_dqn_make_action**

        -  Goal - test if DQN can properly choose an action
        -  Precondition - None
        -  Excepted results - DQN should always return an action from the list of actions

    - **test_dqn_store_memory**

        -  Goal - test if DQN can properly store state, action, next state, reward in the memory 
        -  Precondition - DQN chose at least one action
        -  Excepted results - DQN should properly store state, action, next state, reward in the memory

    - **test_dqn_optimize_model**

        -  Goal - test if DQN will learn in train mode (update weights) and won't learn in test mode
        -  Precondition - None
        -  Excepted results - DQN should be able to learn in train mode. In test mode it shouldn't be able to learn

    - **test_delete_illegal_moves**

        -  Goal - test if DQN won't pick any illegal moves
        -  Precondition - there is at least one illegal move and at least one legal move
        -  Excepted results - DQN should only pick legal moves

    - **test_no_moves**

        -  Goal - test if DQN is able to work properly when there is no state and actions (this happens when game ends)
        -  Precondition - None
        -  Excepted results - DQN should return None

    - **test_restart**

        -  Goal - test if DQN is able to restart properly (number of steps is set to 0)
        -  Precondition - DQN chose at least one action
        -  Excepted results - DQN number of steps should be equal to 0

* **test_logger.py**

    - **test_info**

        -  Goal - test if the logger can log an info log
        -  Precondition - the logger did not store any logs before
        -  Excepted results - the logger should only store an info log

    - **test_log**

        -  Goal - test if the logger can log the log with a given message, level and type
        -  Precondition - the logger did not store any logs before
        -  Excepted results - the logger should only store the log with a given message, level and type
