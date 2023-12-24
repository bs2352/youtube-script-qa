#!/bin/bash

VENV_BIN=".venv/bin/"

cd frontend
if [ ! -d "node_modules" ];
then
    npm install
fi
if [ ! -d "dist" ];
then
    npm run build
fi

# run backend server
cd ..
$VENV_BIN/gunicorn -c gunicorn_config.py restapi:app
