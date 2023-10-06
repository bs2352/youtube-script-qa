from typing import List, Dict, Tuple
import os
import sys
import json
import asyncio
import time

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

        self.summary_file: str = f'{os.environ["SUMMARY_STORE_DIR"]}/{self.vid}'

        self.chain_type: str = 'map_reduce'
        self.llm: LLMType = setup_llm_from_environment()
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


    def _prepare_transcriptions (self) -> None:
        MAXLENGTH = 1000
        OVERLAP_LENGTH = 5
        transcriptions: List[YoutubeTranscriptType] = YouTubeTranscriptApi.get_transcript(video_id=self.vid, languages=["ja", "en", "en-US"])
        self.chunks = divide_transcriptions_into_chunks(
            transcriptions,
            maxlength = MAXLENGTH,
            overlap_length = OVERLAP_LENGTH,
            id_prefix = self.vid
        )


    def run (self) -> Dict[str, int|str|List[str]]:
        self._prepare_transcriptions()

        chain = load_summarize_chain(
            llm=self.llm,
            chain_type=self.chain_type,
            map_prompt=PromptTemplate(template=MAP_PROMPT_TEMPLATE, input_variables=["text"]),
            combine_prompt=PromptTemplate(template=REDUCE_PROMPT_TEMPLATE, input_variables=["text"]),
            verbose=self.debug
        )

        # 簡潔な要約
        tasks = [chain.arun([Document(page_content=chunk.text) for chunk in self.chunks])]
        gather = asyncio.gather(*tasks)
        loop = asyncio.get_event_loop()
        concise_summary = loop.run_until_complete(gather)[0]

        # 詳細な要約
        splited_chunks: List[List[TranscriptChunkModel]] = self._divide_chunks_by_time(5)
        tasks = [chain.arun([Document(page_content=chunk.text) for chunk in chunks]) for chunks in splited_chunks]
        gather = asyncio.gather(*tasks)
        loop = asyncio.get_event_loop()
        detail_summary = loop.run_until_complete(gather)

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


