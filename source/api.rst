API Documentation
====================

------------
Description
------------

This module is designed to be a middleman between translator and RL modules. It is responsible also for initializing the game and translator.
The project is a web application that integrates a game with a translator interface, allowing RL to interact with game in real-time.

------------------------
Main file (`main.py`)
------------------------

The `main.py` file serves as the main entry point for the application. It utilizes the `socketio` library and `aiohttp` for handling asynchronous communication. The core functionality is managed by the `Runner` class, which handles communication with the RL module.

-----------------------------
Runner class (`runner.py`)
-----------------------------

The `runner.py` file contains the `Runner` class, responsible for managing implemented game and restarting game.

The `Runner` class initializes a game and a corresponding translator. The `reset` method is used to reset the game.


--------
Actions
--------

.. _get_response:

^^^^^^^^^^^^^
get_response
^^^^^^^^^^^^^
The **get_response** function, is an asynchronous event handler, it returns the JSON with all the information about the game.
This event is called as a response to :ref:`make_move` call.

Example of JSON response

     .. code-block:: json

        {
            "moves_vector": "moves",
            "game_board": "board",
            "reward": 0,
            "state": "State.ONGOING"
        }

.. _make_move:

^^^^^^^^^^^^^
make_move
^^^^^^^^^^^^^
The **make_move** function, is an asynchronous event handler, it processes move requests from the RL via WebSocket.
It parses JSON data, executes game logic (reset or move), retrieves game information, prepares a response, and emits it to the client.
This action facilitates real-time interaction, updating the game state and providing essential information for possible frontend visualization.

Example of JSON response to make_action call

     .. code-block:: json

        {
            "move": 1
        }

this call will make a move with index 1. And then :ref:`get_response` will be called.

