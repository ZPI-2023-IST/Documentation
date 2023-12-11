What you should know before building your own modules
============================================================

| One of the functionalities of our framework is that you can easily create/modify modules.
| But before you start doing anything it's worth to know more of the modules dependencies.
| It's because modules are dependent on each other and changing modules without much thought could cause the framework to crash.

================================
Modules dependencies
================================

#. Game module
    a. Can be a standalone application
    b. Not dependent on other modules
#. Translator module
    a. Dependent on the Game module and the structure of the model in the RL module
#. API module
    a. Dependent on the Game and Translator modules
    b. Dependent on the RL module
#. RL module
    a. Models are not dependent on other modules
    a. Other functionalities should not be modified
#. Frontend module
    a. Should not be modified
#. Visualization module is dependent on the implementation of the Game modules
    a. Dependent on the Game and Frontend module
    b. API module needs to pass proper data to RL module for Visualization module to work

================================================================
In which order should you build your modules
================================================================

| Let's assume that you want to train a model on your game.
| Accounting for the dependencies described above we recommend implementing modules in this order:

#. Build your own Game module. 
#. Build your own Translator module
#. Build your own API module
#. Build your own model in RL module (optional)
#. Build your own Visualization module (optional)

================================================================
Dockerising modules
================================================================

| After you've build all your modules you can dockerise them to speed up the process of starting the framework
| You don't need to dockerise all of them as just these modules needs to be dockerized by you:

#. API module
#. Visualization module (optional)

| After dockerising your modules you can build a YAML file for Docker Compose
| This file needs to instantiate Docker containers for following modules:

#. API module
#. RL module
#. Frontend module
#. Visualization module (optional)
