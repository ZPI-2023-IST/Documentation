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

TO DO

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2048
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TO DO

--------------------------------------
RL Module
--------------------------------------

All the tests are located `in the RL module tests folder repository <https://github.com/ZPI-2023-IST/RL/tree/master/rl/tests>`_.

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

**test_algorithm_manager.py**

**test_decorator**

| Goal - test if the new algorithm can be properly registered
| Precondition - the new algorithm wasn't registered before
| Excepted results - new algorithm should be registered by algorithm manager

**test_set_algorithm**

| Goal - test if the algorithm can be set as the current algorithm
| Precondition - the algorithm is not set as the current algorithm
| Excepted results - the algorithm should be set as the current algorithm

**test_default_algorithm**

| Goal - test if algorithm manager can set the default algorithm as the current algorithm
| Precondition - None
| Excepted results - the default algorithm should be set as the current algorithm

**test_configure_algorithm**

| Goal - test if algorithm manager can change the parameter values of the current algorithm
| Precondition - None
| Excepted results - the parameter values of the current algorithm should be changed

**test_algorithms.py**

**test_config**

| Goal - test if config can be properly set for the algorithm
| Precondition - None
| Excepted results - config is set properly for the algorithm

**test_random**

| Goal - test if forward method in algorithm works correctly (check if random algorithm can pick an action)
| Precondition - None
| Excepted results - algorithm should always choose the action from the list of actions

**test_registered_algorithms**

| Goal - test if all algorithms registered inherit from the Algorithm class
| Precondition - there is at least one algorithm registered
| Excepted results - all algorithms registered should inherit from the Algorithm class

**test_configurable_params**

| Goal - test if you can get a dictionary of configurable parameters from the algorithm
| Precondition - None
| Excepted results - algorithm returned proper dictionary of configurable parameters

**test_api.py**

Before tests begin we set client that connects to RL module server

**test_config_endpoint**

| Goal - test if you can read and modify model configuration using /config endpoint
| Precondition - None
| Excepted results - model configuration should be properly read and modified

**test_algorithm_update**

| Goal - test if you can create new model with the given dictionary of parameters using /config endpoint
| Precondition - None
| Excepted results - new model should be created and set with the given dictionary of parameters

**test_configurable_params**

| Goal - test if you can get a dictionary of configurable parameters using /config-params endpoint
| Precondition - None
| Excepted results - endpoint should return a proper dictionary of configurable parameters

**test_logs_endpoint**

| Goal - test if you can get a dictionary of logs using /logs endpoint
| Precondition - None
| Excepted results - endpoint should return a proper dictionary of logs

**test_model_endpoint**

| Goal - test if you can get current model parameters using /model endpoint
| Precondition - None
| Excepted results - endpoint should return current model parameters

**test_dqn.py**

Before tests begin we setup DQN algorithm

**test_dqn_make_action**

| Goal - test if DQN can properly choose an action
| Precondition - None
| Excepted results - DQN should always return an action from the list of actions

**test_dqn_store_memory**

| Goal - test if DQN can properly store state, action, next state, reward in the memory 
| Precondition - DQN chose at least one action
| Excepted results - DQN should properly store state, action, next state, reward in the memory

**test_dqn_optimize_model**

| Goal - test if DQN will learn in train mode (update weights) and won't learn in test mode
| Precondition - None
| Excepted results - DQN should be able to learn in train mode. In test mode it shouldn't be able to learn

**test_delete_illegal_moves**

| Goal - test if DQN won't pick any illegal moves
| Precondition - there is at least one illegal move and at least one legal move
| Excepted results - DQN should only pick legal moves

**test_no_moves**

| Goal - test if DQN is able to work properly when there is no state and actions (this happens when game ends)
| Precondition - None
| Excepted results - DQN should return None

**test_restart**

| Goal - test if DQN is able to restart properly (number of steps is set to 0)
| Precondition - DQN chose at least one action
| Excepted results - DQN number of steps should be equal to 0

**test_logger.py**

**test_info**

| Goal - test if the logger can log an info log
| Precondition - the logger did not store any logs before
| Excepted results - the logger should only store an info log

**test_log**

| Goal - test if the logger can log the log with a given message, level and type
| Precondition - the logger did not store any logs before
| Excepted results - the logger should only store the log with a given message, level and type
