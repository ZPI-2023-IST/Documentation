Visualization Model Documentation
===============================

Description
-----------

The Visualization Model is designed to provide a visual representation of game. It is a webapp that
generates a visualization representation of a game based on the provided data which is mostly boards.

This module shouldn't have a logic implementation of a game. It should only be used to generate a visualization.


Endpoints
---------

1. **Generate Visualization Endpoint**

   - **URL:** `/api/visualization`
   - **Method:** `POST`
   - **Request JSON Format:** `example for FreeCell game`


     .. code-block:: json

      [
          {
            "FreeCells": [null, null, null, null],
            "Stack": [null, null, null, null],
            "Board": [
              ["J of h", "8 of s", "5 of d", "Q of h", "9 of c", "K of d"],
              ["2 of c", "8 of c", "3 of h", "4 of h", "4 of d", "7 of c"],
              ["Q of s", "8 of h", "2 of h", "3 of s", "6 of s", "A of h"],
              ["6 of d", "7 of h", "9 of h", "T of s", "J of s", "5 of s"],
              ["A of s", "T of c", "K of h", "6 of h", "Q of c", "K of s", "K of c"],
              ["Q of d", "2 of s", "2 of d", "J of c", "A of d", "9 of d", "9 of s"],
              ["T of h", "A of c", "5 of c", "8 of d", "T of d", "5 of h", "6 of c"],
              ["4 of s", "J of d", "3 of d", "3 of c", "7 of d", "7 of s", "4 of c"]
            ]
          },
          {
            "FreeCells": [null, null, null, "K of d"],
            "Stack": [null, null, null, null],
            "Board": [
              ["J of h", "8 of s", "5 of d", "Q of h", "9 of c"],
              ["2 of c", "8 of c", "3 of h", "4 of h", "4 of d", "7 of c"],
              ["Q of s", "8 of h", "2 of h", "3 of s", "6 of s", "A of h"],
              ["6 of d", "7 of h", "9 of h", "T of s", "J of s", "5 of s"],
              ["A of s", "T of c", "K of h", "6 of h", "Q of c", "K of s", "K of c"],
              ["Q of d", "2 of s", "2 of d", "J of c", "A of d", "9 of d", "9 of s"],
              ["T of h", "A of c", "5 of c", "8 of d", "T of d", "5 of h", "6 of c"],
              ["4 of s", "J of d", "3 of d", "3 of c", "7 of d", "7 of s", "4 of c"]
            ]
          },
            {
            "FreeCells": [null, null, null, "K of d"],
            "Stack": [null, null, null, "7 of c"],
            "Board": [
              ["J of h", "8 of s", "5 of d", "Q of h", "9 of c"],
              ["2 of c", "8 of c", "3 of h", "4 of h", "4 of d"],
              ["Q of s", "8 of h", "2 of h", "3 of s", "6 of s", "A of h"],
              ["6 of d", "7 of h", "9 of h", "T of s", "J of s", "5 of s"],
              ["A of s", "T of c", "K of h", "6 of h", "Q of c", "K of s", "K of c"],
              ["Q of d", "2 of s", "2 of d", "J of c", "A of d", "9 of d", "9 of s"],
              ["T of h", "A of c", "5 of c", "8 of d", "T of d", "5 of h", "6 of c"],
              ["4 of s", "J of d", "3 of d", "3 of c", "7 of d", "7 of s", "4 of c"]
            ]
          },
            ...
        ]
    - **Description:** `This endpoint is used to get a list of boards from frontend module`


   - **Response Format:**

     .. code-block:: json


        {
          "url": "/freecell",
          "data": "JSON_DATA",
        }

    - **Description:** As a response GUI should send a url link which leads to visualization page.

2. **Get Visualization Endpoint**

   - **URL:** `/freecell`
   - **Method:** `GET`
   - **Response Format:** `HTML`
   - **Description:** This url provides a visual representation of posted data.
   - **customization:** This url can and should be modified for a specific game, this is only an example.

