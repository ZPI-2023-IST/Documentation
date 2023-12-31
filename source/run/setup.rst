How to setup the framework
==============================================

=================
Prerequisites
=================

To run the program you need the following software installed:

1. `git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git/>`_
2. `Python 3.10 <https://www.python.org/downloads/release/python-31011/>`_
3. `Node.js <https://docs.npmjs.com/downloading-and-installing-node-js-and-npm/>`_
4. `Docker Compose <https://docs.docker.com/compose/install/>`_ (optional)

================================
Setup
================================

There are two possible ways to setup the framework:

1. :ref:`manual_setup`
2. :ref:`automatic_setup`

| In this tutorial we will setup the framework to train models on `FreeCell <https://github.com/ZPI-2023-IST/FreeCell/>`_ or `2048 <https://github.com/ZPI-2023-IST/2048/>`_ .
| The installment for other games will look similar but there may be slight differences so please read README files before you proceed

.. _manual_setup:

----------------------------------------------------------------------------
Manually setting up entire framework
----------------------------------------------------------------------------

To manually setup the whole framework you'll need to setup these modules:

1. :ref:`api_setup`
2. :ref:`rl_setup`
3. :ref:`frontend_setup`
4. :ref:`visualisation_setup` (optional)

After setting up all the modules you will be able to run the framework any time

.. _api_setup:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To setup API you'll need to perform the following commands in your terminal:

.. code-block:: bash

   git clone https://github.com/ZPI-2023-IST/API
   cd API
   python3 -m venv venv
   pip install -r requirements.txt

   # installing FreeCell and its translator
   pip install git+https://github.com/ZPI-2023-IST/FreeCell.git git+https://github.com/ZPI-2023-IST/Translator.git
   
   # or installing 2048 and its translator
   pip install git+https://github.com/ZPI-2023-IST/2048.git git+https://github.com/ZPI-2023-IST/Translator_2048.git

After performing all the steps run **python main.py** in the terminal. You should see the following screen

.. image:: ../_static/run/setup/api_setup.png
   :width: 600

If you see an error please try again from the start. If that doesn't solve the issue please contact us.

.. _rl_setup:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To setup RL you'll need to perform the following commands in your terminal:

.. code-block:: bash

   git clone https://github.com/ZPI-2023-IST/RL
   cd RL
   python3 -m venv venv
   pip install -e .

| Before your run the RL module check if the config.json file is set up properly

This is the structure of config.json file ::

   {
    "game_address": "http://api",
    "game_port": 5002
   }

Here is the list of parameters you can modify :

#. game_address - if you run the RL module locally set it to *http://localhost*, otherwise leave it as *http://api*
#. game_port - modify only if API module is set up on a different port

NOTE - for this method of setup you need to change game_address to *http://localhost*

After performing all the steps run **python rl/api/main.py** in the terminal. You should see the following screen

.. image:: ../_static/run/setup/rl_setup.png
   :width: 600

If you see an error please try again from the start. If that doesn't solve the issue please contact us.

.. _frontend_setup:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Frontend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To setup Frontend you'll need to perform the following commands in your terminal:

.. code-block:: bash

   git clone https://github.com/ZPI-2023-IST/Frontend
   cd Frontend
   npm install

After performing all the steps run ``npm run dev`` in the terminal. You should see the following screen

.. image:: ../_static/run/setup/frontend_setup.png
   :width: 600

| If you see an error please try again from the start. If that doesn't solve the issue please contact us.
| NOTE - do not click on the link unless you have API and RL modules running

.. _visualisation_setup:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Visualisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To setup Visualisation you'll need to perform the following commands in your terminal:

.. code-block:: bash

   git clone https://github.com/ZPI-2023-IST/FreeCell-GUI
   cd FreeCell-GUI
   npm install

After performing all the steps run ``npm run dev`` in the terminal. You should see the following screen

.. image:: ../_static/run/setup/visualisation_setup.png
   :width: 600

| If you see an error please try again from the start. If that doesn't solve the issue please contact us.
| NOTE - the module should be accessible from Frontend. On itself it won't run

.. _automatic_setup:

----------------------------------------------------------------------------
Using Docker Compose for automatic setup
----------------------------------------------------------------------------

| To make setup easier you can use Docker Compose
| For that to be able to run you'll need to have every module configured to run as a Docker container
| We've shared a script for Docker Compose setup
| To setup the module in this way you need to perform the following commands in your terminal:

.. code-block:: bash

   git clone https://github.com/ZPI-2023-IST/Containers
   cd Containers

   # for FreeCell
   cd Freecell

   # or for 2048
   cd 2048
   
   docker compose up

After performing all the steps you should see the following screen

.. image:: ../_static/run/setup/docker_compose_setup.png
   :width: 600

If you see an error please try again from the start. If that doesn't solve the issue please contact us.
