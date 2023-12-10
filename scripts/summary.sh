#!/bin/bash

CUR_DIR=`pwd`
if [ `basename $CUR_DIR` != "scripts" ]; then
    cd scripts
fi


VID_LIST="../vid.txt" # 動画IDのリスト
SUMMARY_DIR="../data/summaries" # 要約の保存先


main() {
    case $1 in
        "-s" )        summarize ;;
        "summarize" ) summarize ;;
        "summary" )   summary $2;;
        "list" )      list ;;
        * )           list;;
    esac
}


# 動画IDリストの動画を全て要約する
summarize() {
    for vid in `cat $VID_LIST`
    do
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
    for vid in `ls $SUMMARY_DIR`
    do
        summary_file="$SUMMARY_DIR/$vid"
        title=`cat $summary_file | jq | grep title | head -n 1 | cut -d ':' -f2 | sed -e 's/,$//' | sed -e 's/^[ \t]*//' | sed -e 's/^"//' | sed -e 's/"$//'`
        author=`cat $summary_file | jq | grep author | cut -d ':' -f2 | sed -e 's/,$//' | sed -e 's/^[ \t]*//' | sed -e 's/^"//' | sed -e 's/"$//'`
        echo -e "$vid\t$author\t$title"
    done
}

# 要約ディレクトリの要約を順番に表示する
summary() {
    for vid in `ls $SUMMARY_DIR`
    do
        if [ "x${1}" != "x" -a "x${1}" != "x${vid}" ]; then
            continue
        fi
        echo $vid
        # summary_file="$SUMMARY_DIR/$vid"
        # cat $summary_file | jq | less
        cur_dir=`pwd`
        cd ..
        python -c "from yts.summarize import get_summary, YoutubeSummarize; YoutubeSummarize.print(get_summary('${vid}'))" | less
        cd $cur_dir
    done
}


main $1 $2