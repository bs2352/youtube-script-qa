from typing import List, Dict, Optional
import os
import sys
import json
import asyncio
import time

from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

from .types import LLMType, TranscriptChunkModel, YoutubeTranscriptType
from .utils import setup_llm_from_environment, divide_transcriptions_into_chunks, loading_for_async_func


MODE_CONCISE = 0x01
MODE_DETAIL  = 0x02


MAP_PROMPT_TEMPLATE = """以下の内容を重要な情報はできるだけ残して要約してください。:


"{text}"


要約:"""

REDUCE_PROMPT_TEMPLATE = """以下の内容を200字以内の日本語で簡潔に要約してください。:


"{text}"


要約:"""


class YoutubeSummarize:
    def __init__(self,
                 vid: str = "",
                 debug: bool = False
    ) -> None:
        if vid == "":
            raise ValueError("video id is invalid.")

        self.vid: str = vid
        self.debug: bool = debug

        self.summary_file: str = f'{os.environ["SUMMARY_STORE_DIR"]}/{self.vid}'

        self.chain_type: str = 'map_reduce'
        self.llm: LLMType = setup_llm_from_environment()
        self.chain: Optional[BaseCombineDocumentsChain] = None
        self.chunks: List[TranscriptChunkModel] = []

        self.url: str = f'https://www.youtube.com/watch?v={vid}'
        video_info = YouTube(self.url).vid_info["videoDetails"]
        self.title: str = video_info['title']
        self.author: str = video_info['author']
        self.lengthSeconds: int = int(video_info['lengthSeconds'])


    def _debug (self, message: str, end: str = "\n", flush: bool = False) -> None:
        if self.debug is False:
            return
        print(message, end=end, flush=flush)
        return


    @loading_for_async_func
    def run (self, mode:int = MODE_CONCISE|MODE_DETAIL) -> Dict[str, int|str|List[str]]:
        # 準備
        self.chain = self._prepare_summarize_chain()
        self.chunks = self._prepare_transcriptions()

        # 簡潔な要約
        concise_summary = ""
        if mode & MODE_CONCISE > 0:
            concise_summary = self._summarize_concisely()

        # 詳細な要約
        detail_summary: List[str] = []
        if mode & MODE_DETAIL > 0:
            detail_summary = self._summarize_in_detail()

        summary: Dict[str, int|str|List[str]] = {
            "title": self.title,
            "author": self.author,
            "lengthSeconds": self.lengthSeconds,
            "url": self.url,
            "concise": concise_summary,
            "detail": detail_summary,
        }

        if not os.path.isdir(os.path.dirname(self.summary_file)):
            os.makedirs(os.path.dirname(self.summary_file))
        with open(self.summary_file, "w") as f:
            f.write(json.dumps(summary, ensure_ascii=False))

        return summary


    def _prepare_summarize_chain (self) -> BaseCombineDocumentsChain:
        return load_summarize_chain(
            llm=self.llm,
            chain_type=self.chain_type,
            map_prompt=PromptTemplate(template=MAP_PROMPT_TEMPLATE, input_variables=["text"]),
            combine_prompt=PromptTemplate(template=REDUCE_PROMPT_TEMPLATE, input_variables=["text"]),
            verbose=self.debug
        )


    def _prepare_transcriptions (self) -> List[TranscriptChunkModel]:
        MAXLENGTH = 1000
        OVERLAP_LENGTH = 5
        transcriptions: List[YoutubeTranscriptType] = YouTubeTranscriptApi.get_transcript(video_id=self.vid, languages=["ja", "en", "en-US"])
        return divide_transcriptions_into_chunks(
            transcriptions,
            maxlength = MAXLENGTH,
            overlap_length = OVERLAP_LENGTH,
            id_prefix = self.vid
        )


    def _summarize_concisely (self) -> str:
        if self.chain is None:
            return ""
        tasks = [self.chain.arun([Document(page_content=chunk.text) for chunk in self.chunks])]
        gather = asyncio.gather(*tasks)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(gather)[0]


    def _summarize_in_detail (self) -> List[str]:
        if self.chain is None:
            return []
        chunk_groups: List[List[TranscriptChunkModel]] = self._divide_chunks_into_N_groups(5)
        tasks = [
            self.chain.arun([Document(page_content=chunk.text) for chunk in chunks]) for chunks in chunk_groups
        ]
        gather = asyncio.gather(*tasks)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(gather)


    def _divide_chunks_into_N_groups (self, group_num: int = 5) -> List[List[TranscriptChunkModel]]:
        total_time: float = self.chunks[-1].start + self.chunks[-1].duration
        delta: float = total_time // group_num
        groups: List[List[TranscriptChunkModel]] = [[] for _ in range(0, group_num)]
        for chunk in self.chunks:
            idx = int(chunk.start // delta)
            idx = min(idx, group_num - 1)
            groups[idx].append(chunk)
        return [group  for group in groups if len(group) > 0]


    def _divide_chunks_into_groups_by_time_interval (self, interval_minutes: int = 5) -> List[List[TranscriptChunkModel]]:
        total_time: float = self.chunks[-1].start + self.chunks[-1].duration
        delta: int = interval_minutes * 60
        group_num: int = (int(total_time) + 1) // delta
        groups: List[List[TranscriptChunkModel]] = [[] for _ in range(0, group_num)]
        for chunk in self.chunks:
            idx = int(chunk.start // delta)
            idx = min(idx, group_num - 1)
            groups[idx].append(chunk)
        return [group for group in groups if len(group) > 0]

