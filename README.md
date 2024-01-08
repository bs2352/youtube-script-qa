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
  * OPENAI_LLM_MODEL_NAMEは任意で指定する（指定がなければgpt-3,5-turbo）
* Azure OpenAI Serviceを利用する場合
  * OPENAI_API_KEYを削除あるいは#でコメントアウトする。
  * AZURE_OPENAI_API_BASEとAZURE_OPENAI_API_KEYに正しい値を設定する。
  * 環境に合わせて AZURE_LLM_DEPLOYMENT_NAMEとAZURE_LLM_OPENAI_API_VERSIONを指定する。
  * 環境に合わせて AZURE_EMBEDDING_LLM_DEPLOYMENT_NAMEとAZURE_EMBEDDING_OPENAI_API_VERSIONを指定する。
* その他は必要に応じて編集する。
  * INDEX_STORE_DIRはQAを行うためのインデックスを格納するディレクトリを指定する。
  * SUMMARY_STORE_DIRは要約を格納するディレクトリを指定する。

```
$ cp .env.sample .env
$ vim .env
OPENAI_API_KEY=xxx
OPENAI_LLM_MODEL_NAME=gpt-3.5-turbo-1106
AZURE_OPENAI_API_TYPE=azure
AZURE_OPENAI_API_BASE=https://.....
AZURE_OPENAI_API_KEY=xxx
AZURE_LLM_DEPLOYMENT_NAME=gpt-35-turbo
AZURE_LLM_OPENAI_API_VERSION=2023-07-01
AZURE_EMBEDDING_LLM_DEPLOYMENT_NAME=text-embedding-ada-002
AZURE_EMBEDDING_OPENAI_API_VERSION=2023-07-01
INDEX_STORE_DIR=./data/indexes
SUMMARY_STORE_DIR=./data/summaries
LLM_TEMPERATURE=0.0
LLM_REQUEST_TIMEOUT=20
MAX_SUMMARY_LENGTH=400
MAX_SUMMARY_LENGTH_MARGIN=1.0
MAX_KEYWORD=30
MAX_KEYWORD_MARGIN=1.3
MAX_TOPIC_ITEM=15
MAX_TOPIC_ITEM_MARGIN=1.3
MAX_RETRY_COUNT=3
```

## コマンド実行

### QA実行

```
$ python -m yts
Query: <質問を入力>
Answer: <回答が出力される>
```

### 検索実行

```
$ python -m yts --retrieve
Query: <検索クエリを入力>
-- hh:mm:ss (vid [score]) ---
........
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
  -v VID, --vid VID        Youtube動画のID（default:cEynsEWpXdA）
  --source SOURCE  回答を生成する際に参照する検索結果の数を指定する（default:3）
  -d, --detail         回答生成する際に参照した検索結果を表示する
  --debug          デバッグ情報を出力する
  -s, --summary        要約する
  -r, --retrieve     検索する
  -a, --agenda       アジェンダのタイムテーブルを作成する
```


## Webアプリ

### フロントエンド

#### 開発
```
$ cd frontend
$ npm install
$ npm run dev
```

* http://localhost:5173

#### ビルド
```
$ npm run build
```

### バックエンド
#### 実行
```
$ gunicorn -c gunicorn_config.py restapi:app
```

* http://127.0.0.1:8080/
  * こちらでトップ画面にアクセスする場合はフロントエンドをビルドしておく必要がある

#### 開発
* http://127.0.0.1:8080/docs

#### 備考
* vid.txtにVideoIDを1行1エントリ形式で記載しておくとトップ画面で選択できるようになる。開発用の隠し機能。