# 実行するPythonがあるパス
pythonpath = './'

# ワーカー数
workers = 2

# ワーカーのクラスUvicornWorkerを指定 (Uvicornがインストールされている必要がある)
# uvicorn.run(loop='asyncio')を指定するために独自クラスを作成する
# ref. https://www.uvicorn.org/deployment/#gunicorn
from uvicorn.workers import UvicornWorker
class MyUvicornWorker(UvicornWorker):
    CONFIG_KWARGS = {"loop": "asyncio"}
# worker_class = 'uvicorn.workers.UvicornWorker'
worker_class = 'gunicorn_config.MyUvicornWorker'

# IPアドレスとポート
bind = '127.0.0.1:8080'

# プロセスIDを保存するファイル名
pidfile = 'app.pid'

# Pythonアプリに渡す環境変数
# raw_env = ['MODE=PROD']

# デーモン化する場合はTrue
# daemon = True

# エラーログ
# errorlog = './logs/error_log.txt'

# プロセスの名前
proc_name = 'yts_app'

# アクセスログ
# accesslog = './logs/access_log.txt'

reload = True