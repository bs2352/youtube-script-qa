#!/bin/sh

CUR_DIR=`pwd`
if [ `basename $CUR_DIR` != "scripts" ]; then
    cd scripts
fi

VID_LIST="../vid.txt" # 動画IDのリスト

for vid in `cat $VID_LIST`
do
    cur_dir=`pwd`
    cd ..
    echo "# ${vid}"

    # 任意の処理を書く
    python sample.py $vid
    # python -m yts --summary --vid $vid

    cd $cur_dir
    read -p "Press enter to continue" input
done