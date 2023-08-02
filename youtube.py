from typing import Optional, List, TypedDict
from pydantic import BaseModel
import os
from llama_index import download_loader, GPTVectorStoreIndex, Document
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import sys
import logging
from collections import deque
import dotenv


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


INDEX_STORAGE_DIR = "./indexes"
if not os.path.isdir(INDEX_STORAGE_DIR):
    os.mkdir(INDEX_STORAGE_DIR)

dotenv.load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


# video_id = "Tia4YJkNlQ0"
video_id = "cEynsEWpXdA"
if len(sys.argv) > 1:
    video_id = sys.argv[1]


def create_index (video_id: str) -> GPTVectorStoreIndex:
    # YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
    # loader = YoutubeTranscriptReader()
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=cEynsEWpXdA"], languages=["ja"]) # MS2
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=tFgqdHKsOME"], languages=["ja"]) # MS
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=Tia4YJkNlQ0"], languages=["ja"]) # 西園寺
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=oc6RV5c1yd0"])
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=XJRoDEctAwA"])
    # for doc in documents:
    #     print(doc.text, '-------------------')

    # 後々開始時刻も利用したいのでチャンク分割を自作する
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

    index: GPTVectorStoreIndex = GPTVectorStoreIndex.from_documents(documents)

    return index


# インデックス作成
index_dir = f'{INDEX_STORAGE_DIR}/{video_id}'
if not os.path.isdir(index_dir):
    print("creating index ...", end="", flush=True)
    index: GPTVectorStoreIndex = create_index(video_id)
    print("fin", flush=True)
    print(f'save index to {index_dir} ...', end="", flush=True)
    os.mkdir(index_dir)
    index.storage_context.persist(persist_dir=index_dir)
    print("fin", flush=True)
else:
    from llama_index import StorageContext, load_index_from_storage
    print(f'load index from {index_dir} ...', end="", flush=True)
    storage_context: StorageContext = StorageContext.from_defaults(persist_dir=index_dir)
    index: GPTVectorStoreIndex = load_index_from_storage(storage_context) # type: ignore
    print("fin", flush=True)


# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

# より詳細にクエリエンジンを制御したい場合は以下を参照
# https://gpt-index.readthedocs.io/en/v0.6.26/guides/primer/usage_pattern.html
query_engine = index.as_query_engine(similarity_top_k=3)
while True:
    query = input("Query: ").strip()
    if query == "":
        break
    print('Answer: ', end="", flush=True)
    response = query_engine.query(query)
    print(f'{str(response).strip()}\n')
    for node in response.source_nodes:
        id = ""
        for key, val in node.node.dict()["relationships"].items():
            if "node_id" in val.keys():
                id = val["node_id"]
                break
        print(f"--- {id} [{node.score}] ---\n {node.node.get_content()}")
    print("")
