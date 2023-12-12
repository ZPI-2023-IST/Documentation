Build your own model in RL module
==================================

===========================
How to build the model
===========================

To build the model you'll need to inherit the :ref:`Algorithm class from the RL module documentation <algorithm>` and implement all of the methods

======================================================
How to test if the model was implemented correctly
======================================================

#. Model should be created in `algorithms folder <https://github.com/ZPI-2023-IST/RL/blob/master/rl/algorithms>`__.
#. Model needs to be registered by AlgorithmManager
#. Test if the model works on Freecell or 2048 Game
#. Your model should work for every game, not only on specific games

======================================================
Model example (DQN)
======================================================

| Because DQN implementation is long we recommend that you `the implementation on GitHub <https://github.com/ZPI-2023-IST/RL/blob/master/rl/algorithms/learning_algorithms.py>`__.
| The model is based on `DQN tutorial made by PyTorch team <https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html>`__.
| Our implementation of DQN penalizes illegal moves
