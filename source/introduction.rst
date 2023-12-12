What is Framework to Train and Test RL Models?
==========================================================================

===============================
Functionality of the framework
===============================

#. Train and test reinforcement learning model on a given game
#. Import and export the configuration of currently used reinforcement learning model
#. Customise reinforcement learning models by setting the value of the parameters
#. Monitor the training and testing process by seeing logs at current time
#. See statistics of every session
#. See visualization of current game (works only when model is tested)
#. Customise the framework by:
    a. Allowing to connect your own game to the framework
    b. Allowing to implement your own reinforcement learning models
    c. Allowing to connect your own visualization of the game to the framework

============================
Structure of the framework
============================

The framework consists of 4 modules:

#. Game module which consists of:
    a. Game module
    b. Translator module
    c. Api module
#. RL module
#. Frontend module
#. Visualization module (optional)

The structure of the framework and how modules communicate with each other can be seen in the diagram below

.. image:: _static/introduction/framework_structure.jpg

| Replaceable - an element that can be replaced by the user
| Webpage - a web application that offers a user interface that can be run in a browser
| Server - a server that provides some kind of web interface (http, websocket)
| Interface - an element that has certain methods signatures

=================================
How you can use the framework
=================================

TO DO

=======================================
Details of the framework communication
=======================================

TO DO

===================================
Components of the framework
===================================

TO DO
