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

1. Pages - content that is provided to user.
    a. index.js - main page of the frontend. Allows for running, stopping experiments and dowlnoading/uploading models.
    b. config.js - page with configuration form.
    c. logs.js - page with logs collected on backend.
    d. stats.js - page with statistics , containg information about experiments.
    e. visualize.js - page with list of experiments ran in test mode.
2. Components - reusable components that are used in pages.
    a. navbar.js - component that is used to display navigation bar.
    b. layout.js - sets up header for pages.
    c. port.js - parses ports for backend and visualization page.

--------
index.js
--------

The `index.js` file serves as the main page of the Frontend, providing functionality for running and managing RL experiments.

This file imports various components and libraries, including Bootstrap for styling, and defines a `Home` functional component using the Next.js framework and React.js library.

The component includes the following state variables:

- `file`: Represents the selected file for importing the model.
- `mode`: Represents the mode of the experiment (train or test).
- `run`: Indicates whether an experiment is currently running.
- `show`, `showSuccess`, `showError`: Toggle states for displaying alerts and modals.
- `errorMsg`: Stores error messages for display in case of import errors.
- `time`: Represents the elapsed time of the running experiment.
- `steps`: Represents the number of steps completed in the running experiment.

The component fetches data from the backend API to update the state variables and periodically updates the elapsed time

The `handleFileChange`, `handleExport`, `handleImport`, and `handleRun` functions define various actions such as selecting a file, exporting and importing models, and starting/stopping experiments.

The `createRunText` function dynamically generates text for the run button based on the current state.

The component renders a layout, including alerts, buttons for running, importing, and exporting models, and a modal for importing models.

For more details, refer to the actual `index.js` code.

---------
config.js
---------

The configuration.js file serves as the configuration page of the Frontend, allowing users to modify and update RL experiment configurations.

This file is written in JavaScript and utilizes the Next.js framework, React.js library, and Bootstrap for styling. It includes various components and libraries such as Container, Stack, Form, Button, OverlayTrigger, Popover, Modal, Alert, and Layout. Additionally, it makes use of the useState and useEffect hooks for managing component state and handling side effects.

The configuration page interacts with a backend API hosted at "http://localhost" and communicates over a specified port, which is obtained using the Port component. The API endpoints for retrieving configuration options and updating configurations are "/config-params" and "/config," respectively.

The Configuration component manages several state variables to facilitate dynamic interactions and updates on the configuration page:

- algorithm: Represents the currently selected algorithm for configuration.
- config: Holds the current configuration parameters for the selected algorithm.
- serverConfig: Stores the configuration retrieved from the backend for display purposes.
- modConfig: Represents the modified configuration when updating an existing configuration.
- configOptions: Contains configuration options for all algorithms fetched from the backend.
- modConfigOptions: Stores modifiable configuration options used for updating existing configurations.
- show, showSuccess, showError: Toggle states for displaying alerts and modals indicating success or failure.
- errorMsg: Stores error messages received from the backend during configuration updates.
- validated: Represents the validation status of the configuration form.
- modify: Indicates whether the user is modifying an existing configuration.


The configuration page includes the following main functionalities:

1. **Default Configuration Setup:**

The page fetches default configuration options from the backend using the "/config-params" endpoint and sets up the initial state, including the selected algorithm and its corresponding configuration.

2. **Current Configuration Display:**

Users can trigger the display of the current configuration by clicking on the "Current config" button. This information is shown in a popover with details retrieved from the backend using the "/config" endpoint.

3. **Algorithm Selection:**

Users can select different algorithms from a dropdown list, triggering the dynamic loading of configuration options for the selected algorithm. The default algorithm is initially set to "example."

4. **Configuration Modification:**

Users can modify configuration parameters using input fields and checkboxes. The page dynamically generates form elements based on the data retrieved from the backend.

5. **Configuration Submission:**

Users can submit the modified configuration either as a new configuration or to update an existing configuration. The submission triggers a modal confirmation dialog, warning users that updating the configuration will reset all weights of the current model.

6. **Alerts:**

The page displays alerts for successful configuration updates and error messages in case of submission errors.

7. **Tabs for New and Modified Configurations:**

The page includes tabs for creating new configurations and modifying existing configurations. Users can switch between tabs to perform the desired action.
For more details and the actual code implementation, refer to the provided configuration.js file.

-------
logs.js
-------

The `Logs` component in `logs.js` manages the display of logs related to the RL framework. It utilizes the Next.js framework, React.js library, Bootstrap for styling, and additional components such as Card, Button, Layout, Port, useEffect, useState, and ToastContainer from 'react-toastify'. The component communicates with the backend API hosted at "http://localhost" over a specified port obtained using the Port component, and the logs endpoint "/logs".

The Logs component manages several state variables to facilitate dynamic interactions and updates on the log display page:

- filter: Represents the selected log types for display filtering (CONFIG, TEST, TRAIN).
- filterLevel: Represents the selected log levels for display filtering (DEBUG, INFO, WARNING, ERROR, FATAL).
- logs: Holds the log data fetched from the backend.
- fetched: Indicates whether the log data has been successfully fetched from the backend.


The `Logs` component includes the following key features:

1. **Log Filtering:**

Users can filter logs based on their types (`CONFIG`, `TEST`, `TRAIN`) using buttons.
Filter buttons for types include "Config," "Test," and "Train."

2. **Log Level Filtering:**

Users can filter logs based on their levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `FATAL`) using buttons.
Filter buttons for levels include "Debug," "Info," "Warning," "Error," and "Fatal."

3. **Displaying Log Table:**

Logs are displayed in a table format, showing timestamp, type, level, and content.
The content is clickable, triggering a toast notification and copying the log content to the clipboard.

4. **Toast Notifications:**

Toast notifications are used to inform users that log content has been copied to the clipboard.

5. **Fetching Logs:**

The component fetches logs from the backend using the "/logs" endpoint.
Logs are fetched asynchronously using the useEffect hook and displayed once fetched.

6. **Log Styling:**

Logs are displayed as Cards, with different background colors based on their levels (e.g., 'danger' for 'ERROR').
The text color is adjusted for better readability.

7. **Clipboard Copy Functionality:**

Clicking on log content triggers a function that copies the content to the clipboard.
A toast notification confirms the successful copy.

8. **Responsive Design:**

The component is designed to be responsive, with a maximum height for the log display area and scroll functionality.

For more details and the actual code implementation, refer to the provided `logs.js` file.


--------
stats.js
--------

------------
visualize.js
------------

---------
navbar.js
---------

---------
layout.js
---------

-------
port.js
-------
