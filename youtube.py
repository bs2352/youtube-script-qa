from typing import Optional, List, TypedDict
from pydantic import BaseModel
import os
from llama_index import download_loader, GPTVectorStoreIndex, Document, ServiceContext, LLMPredictor, LangchainEmbedding
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import sys
import logging
from collections import deque
import dotenv
import argparse


INDEX_STORAGE_DIR = "./indexes"
DEFAULT_VIDEO_ID = "cEynsEWpXdA" #"Tia4YJkNlQ0" # 西園寺

dotenv.load_dotenv()
service_context: Optional[ServiceContext] = None


class ChunkModel (BaseModel):
    id: str
    text: str
    start: float
    duration: float
    overlap: int


class YoutubeScriptType (TypedDict):
    text: str
    start: float
    duration: float


def setup () -> Optional[ServiceContext]:
    if "OPENAI_API_KEY" in os.environ.keys():
        openai.api_key = os.environ['OPENAI_API_KEY']
        return ServiceContext.from_defaults()

    llm_predictor: LLMPredictor = LLMPredictor(
        llm=AzureOpenAI(
            openai_api_type=os.environ['AZURE_OPENAI_API_TYPE'],
            openai_api_base=os.environ['AZURE_OPENAI_API_BASE'],
            openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
            openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],
            model=os.environ['AZURE_LLM_MODEL_NAME'],
            deployment_name=os.environ['AZURE_LLM_DEPLOYMENT_NAME'],
            client=None
        )
    )
    embedding_llm = LangchainEmbedding(
        OpenAIEmbeddings(
            openai_api_type=os.environ['AZURE_OPENAI_API_TYPE'],
            openai_api_base=os.environ['AZURE_OPENAI_API_BASE'],
            openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
            openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],
            model=os.environ['AZURE_EMBEDDING_LLM_MODEL_NAME'],
            deployment=os.environ['AZURE_EMBEDDING_LLM_DEPLOYMENT_NAME'],
            client=None
        )
    )
    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model=embedding_llm,
    )
   
    return service_context


def create_or_load_index (video_id: str) -> GPTVectorStoreIndex:
    # 既にあれば再利用する
    index_dir = f'{INDEX_STORAGE_DIR}/{video_id}'
    if os.path.isdir(index_dir):
        from llama_index import StorageContext, load_index_from_storage
        print(f'load index from {index_dir} ...', end="", flush=True)
        storage_context: StorageContext = StorageContext.from_defaults(persist_dir=index_dir)
        index: GPTVectorStoreIndex = load_index_from_storage(storage_context, service_context=service_context) # type: ignore
        print("fin", flush=True)
        return index

    # 本にある楽ちんバージョン（使わない）
    # テキストしか取得できず、また後々開始時刻も利用したいのでチャンク分割を自作する
    # YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
    # loader = YoutubeTranscriptReader()
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=cEynsEWpXdA"], languages=["ja"]) # MS2
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=tFgqdHKsOME"], languages=["ja"]) # MS
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=Tia4YJkNlQ0"], languages=["ja"]) # 西園寺
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=oc6RV5c1yd0"])
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=XJRoDEctAwA"])
    # for doc in documents:
    #     print(doc.text, '-------------------')


    def _overlap_chunk (overlaps: deque[YoutubeScriptType]) -> ChunkModel|None:
        if len(overlaps) == 0:
            return None
        new_chunk: ChunkModel = ChunkModel(id="", text="", start=0.0, duration=0.0, overlap=0)
        for s in overlaps:
            new_chunk.text += s['text']
            new_chunk.duration += s['duration']
            if new_chunk.start == 0.0:
                new_chunk.start = s['start']
        return new_chunk


    print("creating index ...", end="", flush=True)

    MAXLENGTH = 300
    OVERLAP_LENGTH = 3
    scripts: List[YoutubeScriptType] = YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=["ja"])
    chunks: List[ChunkModel] = []
    chunk: ChunkModel | None = None
    overlaps: deque[YoutubeScriptType] = deque([])
    for script in scripts:
        if chunk is None:
            chunk = ChunkModel(
                id=f"{video_id}-{script['start']}",
                text=script['text'],
                start=script['start'],
                duration=script['duration'],
                overlap=0
            )
        elif len(chunk.text) - chunk.overlap + len(script["text"]) > MAXLENGTH:
            chunks.append(chunk)
            overlap_chunk: ChunkModel | None = _overlap_chunk(overlaps)
            chunk = ChunkModel(
                id=f'{video_id}-{overlap_chunk.start}',
                text=overlap_chunk.text + script["text"],
                start=overlap_chunk.start,
                duration=overlap_chunk.duration,
                overlap=len(overlap_chunk.text)
            ) if overlap_chunk is not None else ChunkModel(
                id=f'{video_id}-{script["start"]}',
                text=script['text'],
                start=script['start'],
                duration=script['duration'],
                overlap=0
            )
        else:
            chunk.text += script["text"]
            chunk.duration += script["duration"]

        if len(overlaps) < OVERLAP_LENGTH:
            overlaps.append(script)
        else:
            overlaps.popleft()
            overlaps.append(script)
    if chunk is not None:
        chunks.append(chunk)

    # for chunk in chunks:
    #     print(chunk)
    # sys.exit(0)

    documents = [
        Document(text=chunk.text.replace("\n", " "), doc_id=chunk.id) for chunk in chunks
    ]

    index: GPTVectorStoreIndex = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    print("fin", flush=True)

    # ディスクに保存しておく
    print(f'save index to {index_dir} ...', end="", flush=True)
    if not os.path.isdir(INDEX_STORAGE_DIR):
        os.mkdir(INDEX_STORAGE_DIR)
    os.mkdir(index_dir)
    index.storage_context.persist(persist_dir=index_dir)
    print("fin", flush=True)

    return index


def output_source () -> None:
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Youtube動画に対するQAを行うスクリプト')
    parser.add_argument('--vid', default=DEFAULT_VIDEO_ID, help='Youtube動画のID')
    parser.add_argument('--source', default=3, type=int, help='回答を生成する際に参照する検索結果の数')
    parser.add_argument('--detail', action='store_true', help='回答の根拠を表示する')
    args = parser.parse_args()

    video_id = args.vid
    service_context = setup()
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    # インデックス作成
    index: GPTVectorStoreIndex = create_or_load_index(video_id)

    # より詳細にクエリエンジンを制御したい場合は以下を参照
    # https://gpt-index.readthedocs.io/en/v0.6.26/guides/primer/usage_pattern.html
    query_engine = index.as_query_engine(similarity_top_k=args.source)
    while True:
        query = input("Query: ").strip()
        if query == "":
            break
        print('Answer: ', end="", flush=True)
        response = query_engine.query(query)
        print(f'{str(response).strip()}\n')

        if args.detail:
            for node in response.source_nodes:
                id = ""
                for key, val in node.node.dict()["relationships"].items():
                    if "node_id" in val.keys():
                        id = val["node_id"]
                        break
                def _time (id: str) -> str:
                    sec: int = int(float(id.split('-')[1]))
                    s = sec % 60
                    m = (sec // 60) % 60
                    h = (sec // 60) // 60
                    return f'{h}:{m}:{s}'
                print(f"--- {_time(id)} ({id} [{node.score}]) ---\n {node.node.get_content()}")
            print("")
