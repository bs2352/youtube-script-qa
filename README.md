# youtube-script-qa

Youtube動画に対してQAを実行するプログラム

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
  * AZURE_LLM_MODEL_NAMEにはcompletion型のモデルを指定する。
  * INDEX_STORE_DIRはQAを行うためのインデックスを格納するディレクトリを指定する。

```
$ cp .env.sample .env
$ vim .env
OPENAI_API_KEY=xxx
AZURE_OPENAI_API_TYPE=azure
AZURE_OPENAI_API_BASE=xxx
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_API_KEY=xxx
AZURE_LLM_DEPLOYMENT_NAME=text-davinci-003
AZURE_LLM_MODEL_NAME=text-davinci-003
AZURE_EMBEDDING_LLM_DEPLOYMENT_NAME=text-embedding-ada-002
AZURE_EMBEDDING_LLM_MODEL_NAME=text-embedding-ada-002
INDEX_STORE_DIR=./indexe
```

## 実行

### 簡易実行

```
$ python -m yqa
Query: <クエリを入力>
Answer: <回答が出力される>
```

### その他オプション

```
$ python -m yqa --help
usage: __main__.py [-h] [--vid VID] [--source SOURCE] [--detail] [--debug]

Youtube動画に対するQAを行うスクリプト

options:
  -h, --help       show this help message and exit
  --vid VID        Youtube動画のID（default:cEynsEWpXdA）
  --source SOURCE  回答を生成する際に参照する検索結果の数を指定する（default:3）
  --detail         回答生成する際に参照した検索結果を表示する
  --debug          デバッグ情報を出力する
```

