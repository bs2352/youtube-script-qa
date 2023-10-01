from typing import List, Dict
import os
import sys
import json

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

from .types import LLMType, TranscriptChunkModel, YoutubeTranscriptType
from .utils import setup_llm_from_environment, divide_transcriptions_into_chunks


MAP_PROMPT_TEMPLATE = """以下の内容を重要な情報はできるだけ残して要約してください。:


"{text}"


要約:"""

REDUCE_PROMPT_TEMPLATE = """以下の内容を200字以内の日本語で簡潔に要約してください。:


"{text}"


簡潔な要約:"""


class YoutubeSummarize:
    def __init__(self,
                 vid: str = "",
                 debug: bool = False
    ) -> None:
        if vid == "":
            raise ValueError("video id is invalid.")

        self.vid: str = vid
        self.debug: bool = debug
        self.url: str = f'https://www.youtube.com/watch?v={vid}'
        self.title: str = ""

        self.summary_file: str = f'{os.environ["SUMMARY_STORE_DIR"]}/{self.vid}'

        self.chain_type: str = 'map_reduce'
        self.llm: LLMType = setup_llm_from_environment()
        self.chunks: List[TranscriptChunkModel] = []


    def _debug (self, message: str, end: str = "\n", flush: bool = False) -> None:
        if self.debug is False:
            return
        print(message, end=end, flush=flush)
        return


    def prepare (self) -> None:
        self.title = YouTube(self.url).vid_info["videoDetails"]["title"]

        MAXLENGTH = 1000
        OVERLAP_LENGTH = 5
        transcriptions: List[YoutubeTranscriptType] = YouTubeTranscriptApi.get_transcript(video_id=self.vid, languages=["ja", "en", "en-US"])
        self.chunks = divide_transcriptions_into_chunks(
            transcriptions,
            maxlength = MAXLENGTH,
            overlap_length = OVERLAP_LENGTH,
            id_prefix = self.vid
        )


    # def run (self) -> str:
    #     chain = load_summarize_chain(
    #         llm=self.llm,
    #         chain_type=self.chain_type,
    #         map_prompt=PromptTemplate(template=MAP_PROMPT_TEMPLATE, input_variables=["text"]),
    #         combine_prompt=PromptTemplate(template=REDUCE_PROMPT_TEMPLATE, input_variables=["text"]),
    #         verbose=self.debug
    #     )
    #     docs: List[Document] = [Document(page_content=chunk.text) for chunk in self.chunks]
    #     summary = chain.run(docs)

    #     if not os.path.isdir(os.path.dirname(self.summary_file)):
    #         os.makedirs(os.path.dirname(self.summary_file))
    #     with open(self.summary_file, "w") as f:
    #         f.write(summary)

    #     return summary


    def run (self) -> Dict[str, str|List[str]]:
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type=self.chain_type,
            map_prompt=PromptTemplate(template=MAP_PROMPT_TEMPLATE, input_variables=["text"]),
            combine_prompt=PromptTemplate(template=REDUCE_PROMPT_TEMPLATE, input_variables=["text"]),
            verbose=self.debug
        )
        splited_chunks: List[List[TranscriptChunkModel]] = self._divide_chunks_by_time(5)

        detail_summary: List[str] = [
            chain.run([Document(page_content=chunk.text) for chunk in chunks]) for chunks in splited_chunks
        ]
        concise_summary: str = chain.run([Document(page_content=s) for s in detail_summary])

        summary: Dict[str, str|List[str]] = {
            "url": self.url,
            "title": self.title,
            "detail": detail_summary,
            "concise": concise_summary,
        }

        if not os.path.isdir(os.path.dirname(self.summary_file)):
            os.makedirs(os.path.dirname(self.summary_file))
        with open(self.summary_file, "w") as f:
            f.write(json.dumps(summary, ensure_ascii=False))

        return summary


    def _divide_chunks_by_time (self, split_num: int = 5) -> List[List[TranscriptChunkModel]]:
        total_time: float = self.chunks[-1].start + self.chunks[-1].duration
        delta: float = total_time // split_num
        splited_chunks: List[List[TranscriptChunkModel]] = [[] for _ in range(0, split_num)]
        for tc in self.chunks:
            idx = int(tc.start // delta)
            idx = idx if idx < split_num else split_num
            if idx + 1 > len(splited_chunks):
                splited_chunks.append([])
            splited_chunks[idx].append(tc)
        return [sc  for sc in splited_chunks if len(sc) > 0]

