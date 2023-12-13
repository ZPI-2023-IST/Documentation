What is the AI Game Solver Framework?
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

The structure of the framework and how modules communicate with each other can be seen in the UML diagram below

.. image:: _static/introduction/framework_structure.jpg
    :width: 600

| Replaceable - an element that can be replaced by the user
| Webpage - a web application that offers a user interface that can be run in a browser
| Server - a server that provides some kind of web interface (http, websocket)
| Interface - an element that has certain methods signatures

=================================
How you can use the framework
=================================

| The user can perform multiple actions within the system.
| The actions that the user can perform can be seen on the UML use case diagram

.. image:: _static/introduction/use_case_diagram.png
    :width: 600

=======================================
Details of the framework communication
=======================================

| The framework consists of many modules communicating with each other
| The communication of the modules can be seen on the UML sequence diagram below
| In the diagram the user performs the following actions:

#. Sets up all the modules
#. Tests model for some time
#. Sees visualization of one of the games
#. Deactivates all the modules

| NOTE - click on the image if you want to see the diagram more clearly

.. image:: _static/introduction/sequence_diagram.jpg
    :width: 600

===================================
Components of the framework
===================================

| The communication between modules can also be seen on the UML component diagram
| The diagram also shows key components of the framework

.. image:: _static/introduction/component_diagram.jpg
    :width: 600
