from typing import Optional
import dotenv
import argparse
import os
import json

from yts.qa import YoutubeQA
from yts.summarize import YoutubeSummarize
from yts.types import SummaryResultModel

DEFAULT_VIDEO_ID = "cEynsEWpXdA" #"Tia4YJkNlQ0" # 西園寺
DEFAULT_REF_SOURCE = 3

dotenv.load_dotenv()


def qa (args):
    yqa = YoutubeQA(args.vid, args.source, args.detail, True, args.debug)

    # ちょっとサービス（要約はあれば表示する）
    print(f'(Title)\n{yqa.title}')
    if os.path.exists(f'{os.environ["SUMMARY_STORE_DIR"]}/{args.vid}'):
        with open(f'{os.environ["SUMMARY_STORE_DIR"]}/{args.vid}', 'r') as f:
            summary = json.load(f)
        print(f'(Summary)\n{summary["concise"]}')
        print("[Topic]")
        for topic in summary["topic"]:
            print(f'{topic["title"]}')
            print("  ", "\n  ".join(topic["abstract"]), sep="")

    while True:
        query = input("Query: ").strip()
        if query == "":
            break
        answer = yqa.run(query)
        print(f'Answer: {answer}\n')

        if args.detail:
            for score, id, time, source in yqa.get_source():
                print(f"--- {time} ({id} [{score}]) ---\n {source}")
            print("")


def summary (args):
    ys = YoutubeSummarize(args.vid, True, args.debug)
    sm: Optional[SummaryResultModel] = ys.run()
    if sm is None:
        return
    print('[Title]\n', sm.title, '\n')
    print("[Concise Summary]\n", sm.concise, '\n')
    print("[Topic]")
    for topic in sm.topic:
        print(f'{topic.title}')
        print("  ", "\n  ".join(topic.abstract), sep="")
    print("")
    if args.detail is True:
        print('[Detail Summary]')
        for s in sm.detail:
            print(f'・{s}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Youtube動画の視聴を支援するスクリプト')
    parser.add_argument('--vid', default=DEFAULT_VIDEO_ID, help=f'Youtube動画のID（default:{DEFAULT_VIDEO_ID}）')
    parser.add_argument('--source', default=DEFAULT_REF_SOURCE, type=int, help=f'回答を生成する際に参照する検索結果の数を指定する（default:{DEFAULT_REF_SOURCE}）')
    parser.add_argument('--detail', action='store_true', help='回答生成する際に参照した検索結果を表示する')
    parser.add_argument('--debug', action='store_true', help='デバッグ情報を出力する')
    parser.add_argument('--summary', action='store_true', help='要約する')
    args = parser.parse_args()

    if args.summary is False:
        qa(args)

    if args.summary is True:
        summary(args)