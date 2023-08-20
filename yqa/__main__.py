import dotenv
import argparse
from yqa.qa import YoutubeQA

DEFAULT_VIDEO_ID = "cEynsEWpXdA" #"Tia4YJkNlQ0" # 西園寺

dotenv.load_dotenv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Youtube動画に対するQAを行うスクリプト')
    parser.add_argument('--vid', default=DEFAULT_VIDEO_ID, help='Youtube動画のID')
    parser.add_argument('--source', default=3, type=int, help='回答を生成する際に参照する検索結果の数')
    parser.add_argument('--detail', action='store_true', help='回答の根拠を表示する')
    parser.add_argument('--debug', action='store_true', help='デバッグ情報を出力する')
    args = parser.parse_args()

    yqa = YoutubeQA(args.vid, args.source, args.detail, args.debug)
    yqa.prepare_query()

    while True:
        query = input("Query: ").strip()
        if query == "":
            break
        print('Answer: ', end="", flush=True)
        answer = yqa.run_query(query)
        print(f'{answer}\n')

        if args.detail:
            for score, id, time, source in yqa.get_source():
                print(f"--- {time} ({id} [{score}]) ---\n {source}")
            print("")
