Build your own Visualization module
=====================================

===========================
How to build the module
===========================

You can build Visualization module in any way you like. The only requirement is that the module needs to be able to connect to Frontend module

======================================================
How to test if the module was implemented correctly
======================================================

Your module needs to properly read data that API module passes through RL and Frontend module
Visualization module is not required to implement logic of the game. It should at least visualize the boards from the Game module.
Unless you decide to bind your visualization with other modules without using Docker, it is essential to include a Dockerfile in your visualization project.
Visualization page should work correctly and show visualizations of all the games

======================================================
Visualization examples
======================================================

See :ref:`vmd` for FreeCell and 2048 visualization examples