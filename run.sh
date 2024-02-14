#!/bin/bash

ENV_SRC="${HOME}/study/yts/.env"
VID_SRC="${HOME}/study/yts/vid.txt"
if [ ! -e ".env" -a -e $ENV_SRC ];
then
    ln $ENV_SRC
fi
if [ ! -e "vid.txt" -a -e $VID_SRC ];
then
    ln $VID_SRC
fi

source .env

VENV_BIN=".venv/bin/"
if [ ! -e ${VENV_BIN} ];
then
    pipenv sync
    echo "Setup python venv done."
    echo "Please close and open vscode, and run this script."
    exit 0
fi

main() {
    case $1 in
        "-f" )  run_frontend;;
        "-i" )  remove_data;;
        "-b" )  run_backend build;;
        * )     run_backend;;
    esac
}

run_frontend () {
    prepare_frontend
    cd frontend
    npm run dev
}

run_backend () {
    if [ "x${1}" = "xbuild" ];
    then
        remove_static_file
        build_frontend
    fi
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

remove_static_file () {
    cd frontend
    rm -rf dist
    cd ..
}

remove_data () {
    rm -rf $INDEX_STORE_DIR
    rm -rf $SUMMARY_STORE_DIR
}

main $1 $2