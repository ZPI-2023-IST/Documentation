Dockerising modules
==================================

| This tutorial isn't meant to describe how to write Docker Containers
| This tutorial is meant to tell how you customise Docker Containers and YAML files that we've created
| Each file that we've created you can copy to your modules and modify according to your needs

================================================================
API Docker Container example (Freecell)
================================================================

Let's take a look at `API module <https://github.com/ZPI-2023-IST/API>`__ Dockerfile::

    # Use an official Python runtime as a parent image
    FROM python:3.10-slim

    # Set the working directory to /app
    WORKDIR /app

    # Copy the current directory contents into the container at /app
    COPY . /app

    # Repos urls
    ARG GAME="git+https://github.com/ZPI-2023-IST/FreeCell.git"
    ARG TRANSLATOR="git+https://github.com/ZPI-2023-IST/Translator.git"

    # Install git
    RUN apt-get update && \
        apt-get install -y git

    # Downloand game and translator from git

    RUN pip3 install --upgrade pip

    RUN pip3 install $GAME
    RUN pip3 install $TRANSLATOR

    # Install any needed packages specified in requirements.txt
    RUN pip install --no-cache-dir -r requirements.txt

    # Make port 5002 available to the world outside this container
    EXPOSE 5002

    # Run app.py when the container launches
    CMD ["python3", "main.py"]

| You can modify this script by swapping GAME and TRANSLATOR args with links to your repositories
| Than the script should properly load your Game and Translator modules

================================================================
Visualization Docker Container example (Freecell)
================================================================

Let's take a look at `Visualisation module <https://github.com/ZPI-2023-IST/FreeCell-GUI>`__ Dockerfile::

    # Use an official Node.js runtime as a base image
    FROM node:18-alpine

    # Set the working directory inside the container
    WORKDIR /usr/src/app

    # Copy package.json and package-lock.json to the working directory
    COPY package*.json ./

    RUN npm cache clean --force
    # Install project dependencies

    RUN npm install

    # Copy the rest of the application code to the working directory
    COPY . .

    # Expose the port that your Next.js app will run on
    EXPOSE 5005

    # Command to run your Next.js app
    CMD ["npm", "run", "dev"]

| If you are using Next.js than no modification is required in the script

================================================================
YAML file for Docker Compose example (Freecell)
================================================================

Let's take a look at `YAML file for Docker Compose <https://github.com/ZPI-2023-IST/Containers>`__ ::

    version: '3'

    services:

        api:
            build: "https://github.com/ZPI-2023-IST/API.git"
            ports:
              - "5002:5002"
            networks:
              - app-network

        frontend:
            build: "https://github.com/ZPI-2023-IST/Frontend.git"
            ports:
              - "3000:3000"
            networks:
              - app-network

        rl:
            build: "https://github.com/ZPI-2023-IST/RL.git"
            ports:
              - "5000:5000"
            networks:
              - app-network
            depends_on:
              - api

        freecell-gui:
            build: "https://github.com/ZPI-2023-IST/FreeCell-GUI.git"
            ports:
              - "5005:5005"
            networks:
              - app-network
            depends_on:
              - frontend

    networks:
        app-network:
            driver: "bridge"

| You can modify this script by swapping api and freecell-gui with your API and Visualization module
| You only need to change the link to build with the link to your repositories
| NOTE - both API and Visualization modules need to have a Dockerfile for Docker Compose to work
| Than the script should properly load your API and Visualisation modules
