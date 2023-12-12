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

All the tests are located `in RL module tests folder repository <https://github.com/ZPI-2023-IST/RL/tree/master/rl/tests>`_.

**How to run tests**

To run tests you'll need to:

#. :ref:`Setup RL module <rl_setup>`
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
