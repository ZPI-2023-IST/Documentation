Frontend Documentation
==============================================

=================
Description
=================

Frontend provides user web interface for interacting with RL framework backend.
It is build using Next.js framework and React.js library. 

=================
Functionality
=================

#. Provides interface for running and stopping RL experiments.
#. Provides oncfirguration form, that allows to set up experiment parameters.
#. Provides interface for monitoring experiment progress.
#. Provides site with list of experiments ran in test mode.
#. Provides sites with logs, collected on backend.

==================================
Structure of the Frontend
==================================

Project contains:
#. Pages - content that is provided to user.
#. Componenets - reusable components, that are used in pages.

-----------------
Pages
-----------------

#. index.js - main page of the frontend. Allows for running, stopping experiments and dowlnoading/uploading models.
#. config.js - page with configuration form.
#. logs.js - page with logs collected on backend.
#. stats.js - page with statistics , containg information about experiments.
#. visualize.js - page with list of experiments ran in test mode.

^^^^^^^
index.js
^^^^^^^

^^^^^^^
config.js
^^^^^^^

^^^^^^^
logs.js
^^^^^^^

^^^^^^^
stats.js
^^^^^^^

^^^^^^^^^^^
visualize.js
^^^^^^^^^^^

-----------------
Components
-----------------

#. navbar.js - component that is used to display navigation bar.
#. layout.js - sets up header for pages.
#. port.js - parses ports for backend and visualization page.

^^^^^^^^
navbar.js
^^^^^^^^

^^^^^^^^
layout.js
^^^^^^^^

^^^^^^^^
port.js
^^^^^^^^
