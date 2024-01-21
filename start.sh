#!/bin/bash

VENV_BIN=".venv/bin/"


main() {
    case $1 in
        "-f" )        run_frontend;;
        "-i" )        remove_data;;
        * )           run_backend;;
    esac
}

run_frontend () {
    prepare_frontend
    cd frontend
    npm run dev
}

run_backend () {
    prepare_frontend
    build_frontend
    $VENV_BIN/gunicorn -c gunicorn_config.py restapi:app
}

prepare_frontend() {
    cd frontend
    if [ ! -d "node_modules" ];
    then
        npm install
    fi
    cd ..
}

build_frontend () {
    cd frontend
    if [ ! -d "dist" ];
    then
        npm run build
    fi
    cd ..
}

remove_data () {
    rm -rf data
}

main $1