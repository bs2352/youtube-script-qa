from typing import List, Optional, Dict
import os
import sys
import json
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

from .types import LLMType, TranscriptChunkModel, YoutubeTranscriptType
from .utils import setup_llm_from_environment, divide_transcriptions_into_chunks
from .types import SummaryResultType, TopicType


MODE_CONCISE = 0x01
MODE_DETAIL  = 0x02
MODE_TOPIC   = 0x04
MODE_ALL     = 0xff


MAP_PROMPT_TEMPLATE = """以下の内容を重要な情報はできるだけ残して要約してください。:


"{text}"


要約:"""

REDUCE_PROMPT_TEMPLATE = """以下の内容を全体を網羅して200字以内の日本語で簡潔に要約してください。:


"{text}"


要約:"""

CONCISELY_PROMPT_TEMPLATE = """以下に記載する動画のタイトルと内容から質問に回答してください。

タイトル：
{title}

内容：
{content}

質問：
この動画の内容を全体を網羅して簡潔に要約してください。

回答：
"""
CONCISELY_PROMPT_TEMPLATE_VARIABLES = ["title", "content"]

TOPIC_PROMPT_TEMPLATE = \
"""I am creating an agenda for Youtube videos.
Below are notes on creating an agenda, as well as video title and abstract.
Please follow the instructions carefully and create an agenda from the title and abstract.

Notes:
- Please create an agenda that covers the entire content of the video.
- Your agenda should include headings and a summary for each heading.
- Please include important keywords in the heading and summary whenever possible.
- Please assign each heading a sequential number such as 1, 2, 3.
- Please keep each heading as concise as possible.
- Please add a "-" to the beginning of each summary and output it as bullet points.
- Please create the summary as a subtitle, not as a sentence.
- Please keep each summary as concise as possible.
- Please create the agenda in Japanese.

title:
{title}

abstract:
{abstract}

agenda:

"""
TOPIC_PROMPT_TEMPLATE_VARIABLES = ["title", "abstract"]


class YoutubeSummarize:
    def __init__(
        self,
        vid: str = "",
        loading: bool = False,
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

        self.loading: bool = loading
        self.loading_canceled: bool = False


    def _debug (self, message: str, end: str = "\n", flush: bool = False) -> None:
        if self.debug is False:
            return
        print(message, end=end, flush=flush)
        return


    def run (self, mode:int = MODE_ALL) -> Optional[SummaryResultType]:
        if self.loading is True:
            return self._run_with_loading(mode)
        return self._run(mode)


    def _run_with_loading (self, mode:int = MODE_CONCISE|MODE_DETAIL) -> Optional[SummaryResultType]:
        summary: Optional[SummaryResultType] = None
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_loading = executor.submit(self._loading)
            try:
                summary = self._run(mode)
            except:
                pass
            self.loading_canceled = True
            while future_loading.done() is False:
                time.sleep(0.5)
        return summary


    def _run (self, mode:int = MODE_ALL) -> SummaryResultType:
        # mode調整
        if mode & MODE_TOPIC > 0:
            mode |= MODE_CONCISE
        if mode & MODE_CONCISE > 0:
            mode |= MODE_DETAIL

        # 準備
        self.chain = self._prepare_summarize_chain()
        self.chunks = self._prepare_transcriptions()

        # 詳細な要約
        detail_summary: List[str] = []
        if mode & MODE_DETAIL > 0:
            detail_summary = self._summarize_in_detail()

        # 簡潔な要約
        concise_summary = ""
        if mode & MODE_CONCISE > 0:
            # concise_summary = self._summarize_concisely()
            concise_summary = self._summarize_concisely(detail_summary)

        # トピック抽出
        topic: List[TopicType] = []
        if mode & MODE_TOPIC > 0:
            topic = self._extract_topic(detail_summary)

        summary: SummaryResultType = {
            "title": self.title,
            "author": self.author,
            "lengthSeconds": self.lengthSeconds,
            "url": self.url,
            "concise": concise_summary,
            "detail": detail_summary,
            "topic": topic,
        }

        # 全モードの場合のみ保存する
        if mode == MODE_ALL:
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


    def _summarize_concisely (self, detail_summary: List[str]) -> str:
        prompt = PromptTemplate(
            template=CONCISELY_PROMPT_TEMPLATE,
            input_variables=CONCISELY_PROMPT_TEMPLATE_VARIABLES,
        )
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt
        )
        args: Dict[str, str] = {
            "title": self.title,
            "content": "\n".join(detail_summary),
        }
        result: str = chain.run(**args)
        return result


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


    def _extract_topic (self, summary: List[str] = []) -> List[TopicType]:
        prompt = PromptTemplate(
            template=TOPIC_PROMPT_TEMPLATE,
            input_variables=TOPIC_PROMPT_TEMPLATE_VARIABLES,
        )
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt
        )
        args: Dict[str, str] = {
            "title": self.title,
            "abstract": "\n".join(summary) if len(summary) > 0 else "",
        }
        result: str = chain.run(**args)
        topic: List[TopicType] = self._parse_topic(result)
        return topic


    def _parse_topic (self, topic_string: str) -> List[TopicType]:
        topics: List[TopicType] = []
        topic: TopicType = {"title": "", "abstract": []}
        for line in topic_string.split("\n"):
            line = line.strip()
            if line == "":
                continue
            if line[0].isdigit():
                if topic["title"] != "":
                    topics.append(topic)
                topic = {"title": line, "abstract": []}
                continue
            if line[0] == "-":
                topic["abstract"].append(line)
        return topics


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


    def _loading (self):
        chars = [
            '/', '―', '\\', '|', '/', '―', '\\', '|', '😍',
            '/', '―', '\\', '|', '/', '―', '\\', '|', '🤪',
            '/', '―', '\\', '|', '/', '―', '\\', '|', '😎',
        ]
        self.loading_canceled = False
        i = 0
        while i >= 0:
            i %= len(chars)
            sys.stdout.write("\033[2K\033[G %s " % chars[i])
            sys.stdout.flush()
            time.sleep(1.0)
            i += 1
            if self.loading_canceled is True:
                break
        sys.stdout.write("\033[2K\033[G")
        sys.stdout.flush()
        return


def get_summary (vid: str, mode: int = MODE_DETAIL) -> str:
    if mode != MODE_CONCISE and mode != MODE_DETAIL and mode != MODE_TOPIC:
        raise ValueError("mode is invalid.")

    summary: Optional[SummaryResultType] = None
    summary_file: str = f'{os.environ["SUMMARY_STORE_DIR"]}/{vid}'
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    else:
        # summary = YoutubeSummarize(vid, debug=True).run(mode=mode)
        summary = YoutubeSummarize(vid).run(mode=mode)

    if summary is None:
        return ""
    if mode == MODE_CONCISE:
        return summary['concise'].strip()
    if mode == MODE_TOPIC:
        # return "\n".join(summary["topic"])
        return ""
    return "\n".join(summary["detail"])
