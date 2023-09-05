# youtube-supporter

Youtube動画の視聴を支援するプログラム

## 準備

### Python

```
$ pipenv shell
$ pipenv install
```

### 実行環境

* .envを作成して実行環境に合わせて以下のように編集する。
* OpenAIを利用する場合
  * OPENAI_API_KEYに正しい値を設定する。
* Azure OpenAI Serviceを利用する場合
  * OPENAI_API_KEYを削除あるいは#でコメントアウトする。
  * AZURE_OPENAI_API_BASEとAZURE_OPENAI_API_KEYに正しい値を設定する。
* その他は必要に応じて編集する。
  * INDEX_STORE_DIRはQAを行うためのインデックスを格納するディレクトリを指定する。
  * SUMMARY_STORE_DIRは要約を格納するディレクトリを指定する。

```
$ cp .env.sample .env
$ vim .env
OPENAI_API_KEY=xxx
OPENAI_LLM_MODEL_NAME=gpt-3.5-turbo-0613
AZURE_OPENAI_API_TYPE=azure
AZURE_OPENAI_API_BASE=xxx
AZURE_OPENAI_API_KEY=xxx
AZURE_LLM_DEPLOYMENT_NAME=gpt-35-turbo
AZURE_LLM_OPENAI_API_VERSION=2023-05-15
AZURE_EMBEDDING_LLM_DEPLOYMENT_NAME=text-embedding-ada-002
AZURE_EMBEDDING_OPENAI_API_VERSION=2023-05-15
INDEX_STORE_DIR=./indexes
SUMMARY_STORE_DIR=./summaries
LLM_TEMPERATURE=0.0
```

## 実行

### QA実行

```
$ python -m yts
Query: <クエリを入力>
Answer: <回答が出力される>
```

### 要約実行

```
$ python -m yts --summary
```


### その他オプション

```
$ python -m yts --help
usage: __main__.py [-h] [--vid VID] [--source SOURCE] [--detail] [--debug] [--summary]

Youtube動画の視聴を支援するスクリプト

options:
  -h, --help       show this help message and exit
  --vid VID        Youtube動画のID（default:cEynsEWpXdA）
  --source SOURCE  回答を生成する際に参照する検索結果の数を指定する（default:3）
  --detail         回答生成する際に参照した検索結果を表示する
  --debug          デバッグ情報を出力する
  --summary        要約する
```

