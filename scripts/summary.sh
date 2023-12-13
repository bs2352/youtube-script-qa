#!/bin/bash

CUR_DIR=`pwd`
if [ `basename $CUR_DIR` != "scripts" ]; then
    cd scripts
fi

CHECK=`which python | grep .venv`
if [ "x${CHECK}" = "x" ]; then
    source ../.venv/bin/activate
fi

source ../.env
export SUMMARY_STORE_DIR="../${SUMMARY_STORE_DIR}" # 要約の保存ディレクトリ
VID_LIST="../vid.txt" # 動画IDのリスト


main() {
    case $1 in
        "-s" )        summarize $2;;
        "summarize" ) summarize $2;;
        "summary" )   summary $2;;
        "list" )      list ;;
        * )           list;;
    esac
}


# 動画IDリストの動画を全て要約する
summarize() {
    for vid in `cat $VID_LIST`
    do
        if [ "x${1}" != "x" -a "x${1}" != "x${vid}" ]; then
            continue
        fi
        cur_dir=`pwd`
        cd ..
        echo "# ${vid}"
        time python -m yts --summary --vid $vid
        echo ""
        cd $cur_dir
        sleep 7
    done
}


# 要約ディレクトリにある全ての要約を一覧表示する
list() {
    for vid in `ls $SUMMARY_STORE_DIR`
    do
        summary_file="$SUMMARY_STORE_DIR/$vid"
        title=`cat $summary_file | jq | grep title | head -n 1 | cut -d ':' -f2 | sed -e 's/,$//' | sed -e 's/^[ \t]*//' | sed -e 's/^"//' | sed -e 's/"$//'`
        author=`cat $summary_file | jq | grep author | cut -d ':' -f2 | sed -e 's/,$//' | sed -e 's/^[ \t]*//' | sed -e 's/^"//' | sed -e 's/"$//'`
        echo -e "$vid\t$author\t$title"
    done
}

# 要約ディレクトリの要約を順番に表示する
summary() {
    for vid in `ls $SUMMARY_STORE_DIR`
    do
        if [ "x${1}" != "x" -a "x${1}" != "x${vid}" ]; then
            continue
        fi
        echo $vid
        # summary_file="$SUMMARY_STORE_DIR/$vid"
        # cat $summary_file | jq | less
        cur_dir=`pwd`
        cd ..
        python -c "from yts.summarize import YoutubeSummarize; YoutubeSummarize.print(YoutubeSummarize.summary('${vid}'))" | less
        cd $cur_dir
    done
}


main $1 $2