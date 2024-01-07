from typing import Optional
import argparse
import os
import json
import sys

from yts.qa import YoutubeQA
from yts.summarize import YoutubeSummarize, MODE_ALL, MODE_DETAIL
from yts.types import SummaryResultModel, SourceModel
from yts.tools import YoutubeAgendaTimeTable


DEFAULT_VIDEO_ID = "cEynsEWpXdA" #"Tia4YJkNlQ0" # 西園寺
DEFAULT_REF_SOURCE = 3


def qa (args):
    yqa = YoutubeQA(args.vid, args.source, args.detail, not args.debug, args.debug)

    # ちょっとサービス（要約はあれば表示する）
    if os.path.exists(f'{os.environ["SUMMARY_STORE_DIR"]}/{args.vid}'):
        with open(f'{os.environ["SUMMARY_STORE_DIR"]}/{args.vid}', 'r') as f:
            summary: SummaryResultModel = SummaryResultModel(**json.load(f))
        YoutubeSummarize.print(summary, MODE_ALL & ~MODE_DETAIL)
    else:
        print(f'[Title]\n{yqa.title}\n')

    while True:
        query = input("Query: ").strip()
        if query == "":
            break
        answer = yqa.run(query)
        print(f'Answer: {answer}\n')

        if args.detail is False:
            continue

        for source in yqa.get_source():
            print(f"--- {source.time} ({source.id} [{source.score}]) ---\n {source.source}")
        print("")


def retrieve (args):
    yqa = YoutubeQA(args.vid, args.source, args.detail, not args.debug, args.debug)

    # ちょっとサービス（要約はあれば表示する）
    if os.path.exists(f'{os.environ["SUMMARY_STORE_DIR"]}/{args.vid}'):
        with open(f'{os.environ["SUMMARY_STORE_DIR"]}/{args.vid}', 'r') as f:
            summary: SummaryResultModel = SummaryResultModel(**json.load(f))
        YoutubeSummarize.print(summary, MODE_ALL & ~MODE_DETAIL)
    else:
        print(f'[Title]\n{yqa.title}\n')

    while True:
        query = input("Query: ").strip()
        if query == "":
            break
        results = yqa.retrieve(query)
        print("Results: ")
        for result in results:
            print(f"--- {result.time} ({result.id} [{result.score}]) ---\n {result.source}")
        print("")


def summary (args):
    ys = YoutubeSummarize(args.vid, not args.debug, args.debug)
    sm: Optional[SummaryResultModel] = ys.run()
    YoutubeSummarize.print(sm, MODE_ALL if args.detail else MODE_ALL & ~MODE_DETAIL)


def agenda (args):
    ys = YoutubeSummarize(args.vid, not args.debug, args.debug)
    sm: Optional[SummaryResultModel] = ys.run()
    sm = YoutubeAgendaTimeTable.make(vid=args.vid, summary=sm, store=True)
    YoutubeAgendaTimeTable.print(sm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Youtube動画の視聴を支援するスクリプト')
    parser.add_argument('-v', '--vid', default=DEFAULT_VIDEO_ID, help=f'Youtube動画のID（default:{DEFAULT_VIDEO_ID}）')
    parser.add_argument('--source', default=DEFAULT_REF_SOURCE, type=int, help=f'回答を生成する際に参照する検索結果の数を指定する（default:{DEFAULT_REF_SOURCE}）')
    parser.add_argument('-d', '--detail', action='store_true', help='回答生成する際に参照した検索結果を表示する')
    parser.add_argument('--debug', action='store_true', help='デバッグ情報を出力する')
    parser.add_argument('-s', '--summary', action='store_true', help='要約する')
    parser.add_argument('-r', '--retrieve', action='store_true', help='検索する')
    parser.add_argument('-a', '--agenda', action='store_true', help='アジェンダのタイムテーブルを作成する')
    args = parser.parse_args()

    if args.summary is True:
        summary(args)
        sys.exit(0)

    if args.retrieve is True:
        retrieve(args)
        sys.exit(0)

    if args.agenda is True:
        agenda(args)
        sys.exit(0)

    qa(args)
    sys.exit(0)

