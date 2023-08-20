from typing import Optional, List, TypedDict, Generator, Tuple
from pydantic import BaseModel
import os
from llama_index import download_loader, GPTVectorStoreIndex, Document, ServiceContext, LLMPredictor, LangchainEmbedding
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.schema import NodeWithScore
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import sys
import logging
from collections import deque


DEFAULT_VIDEO_ID = "cEynsEWpXdA" #"Tia4YJkNlQ0" # 西園寺


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


class YoutubeQA:
    def __init__(self,
                 vid: str = DEFAULT_VIDEO_ID,
                 ref_sources: int = 3,
                 detail: bool = False,
                 debug: bool = False
    ) -> None:
        self.vid: str = vid
        self.ref_source: int = ref_sources
        self.detail: bool = detail
        self.debug: bool = debug

        self.index_dir: str = f'{os.environ["INDEX_STORE_DIR"]}/{self.vid}'
        self.service_context: ServiceContext = self._setup_llm()

        self.index: Optional[GPTVectorStoreIndex] = None
        self.query_response: Optional[RESPONSE_TYPE] = None


    def _setup_llm (self) -> ServiceContext:
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
    

    def _debug (self, message: str, end: str = "\n", flush: bool = False) -> None:
        if self.debug is False:
            return
        print(message, end=end, flush=flush)
        return


    def prepare_query (self) -> None:
        self.index = self._load_index() if os.path.isdir(self.index_dir) else \
                     self._create_index()


    def _load_index (self) -> GPTVectorStoreIndex:
            from llama_index import StorageContext, load_index_from_storage
            self._debug(f'load index from {self.index_dir} ...', end="", flush=True)
            storage_context: StorageContext = StorageContext.from_defaults(persist_dir=self.index_dir)
            index: GPTVectorStoreIndex = load_index_from_storage(storage_context, service_context=self.service_context) # type: ignore
            self._debug("fin", flush=True)
            return index


    def _create_index (self) -> GPTVectorStoreIndex:

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


        self._debug("creating index ...", end="", flush=True)

        MAXLENGTH = 300
        OVERLAP_LENGTH = 3
        scripts: List[YoutubeScriptType] = YouTubeTranscriptApi.get_transcript(video_id=self.vid, languages=["ja"])
        chunks: List[ChunkModel] = []
        chunk: ChunkModel | None = None
        overlaps: deque[YoutubeScriptType] = deque([])
        for script in scripts:
            if chunk is None:
                chunk = ChunkModel(
                    id=f"{self.vid}-{script['start']}",
                    text=script['text'],
                    start=script['start'],
                    duration=script['duration'],
                    overlap=0
                )
            elif len(chunk.text) - chunk.overlap + len(script["text"]) > MAXLENGTH:
                chunks.append(chunk)
                overlap_chunk: ChunkModel | None = _overlap_chunk(overlaps)
                chunk = ChunkModel(
                    id=f'{self.vid}-{overlap_chunk.start}',
                    text=overlap_chunk.text + script["text"],
                    start=overlap_chunk.start,
                    duration=overlap_chunk.duration,
                    overlap=len(overlap_chunk.text)
                ) if overlap_chunk is not None else ChunkModel(
                    id=f'{self.vid}-{script["start"]}',
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

        index: GPTVectorStoreIndex = GPTVectorStoreIndex.from_documents(documents, service_context=self.service_context)

        self._debug("fin", flush=True)

        # ディスクに保存しておく
        self._debug(f'save index to {self.index_dir} ...', end="", flush=True)
        if not os.path.isdir(self.index_dir):
            os.makedirs(self.index_dir)
        index.storage_context.persist(persist_dir=self.index_dir)
        self._debug("fin", flush=True)

        return index


    def run_query (self, query: str) -> str:
        if self.index is None:
            return ""

        # より詳細にクエリエンジンを制御したい場合は以下を参照
        # https://gpt-index.readthedocs.io/en/v0.6.26/guides/primer/usage_pattern.html
        query_engine: BaseQueryEngine = self.index.as_query_engine(similarity_top_k=self.ref_source)
        response: RESPONSE_TYPE = query_engine.query(query)
        self.query_response = response

        return str(self.query_response).strip()
    

    def get_source (self) -> Generator[Tuple[float, str, str, str], None, None]:
        if self.query_response is None:
            return None

        def _id2time (id: str) -> str:
            sec: int = int(float(id.split('-')[1]))
            s = sec % 60
            m = (sec // 60) % 60
            h = (sec // 60) // 60
            return f'{h}:{m}:{s}'
        
        # 暫定版（もっと良い取り方があるはず）
        def _get_id (node: NodeWithScore) -> str:
            id = ""
            for _, val in node.node.dict()["relationships"].items():
                if "node_id" in val.keys():
                    id = val["node_id"]
            return id

        for node in self.query_response.source_nodes:
            id = _get_id(node)
            score: float = node.score if node.score is not None else 0.0
            source: str = node.node.get_content()
            yield score, id, _id2time(id), source


