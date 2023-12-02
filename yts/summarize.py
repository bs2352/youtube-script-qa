from typing import List, Optional, Dict, Tuple
import os
import sys
import json
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError

from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube

from .types import LLMType, TranscriptChunkModel, YoutubeTranscriptType
from .utils import setup_llm_from_environment, divide_transcriptions_into_chunks
from .types import SummaryResultModel, TopicModel


MODE_CONCISE = 0x01
MODE_DETAIL  = 0x02
MODE_TOPIC   = 0x04
MODE_ALL     = MODE_CONCISE | MODE_DETAIL | MODE_TOPIC

SPECIFY_SUMMARY_MAX_LENGTH = os.getenv("SPECIFY_SUMMARY_MAX_LENGTH", "false") == "true"
MAX_CONCISE_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH", "400"))
MAX_LENGTH_MARGIN_MULTIPLIER = float(os.getenv("MAX_SUMMARY_LENGTH_MARGIN", "1.0"))
MAX_TOPIC_ITEMS = 15
MAX_RETRY_COUNT = 5
RETRY_INTERVAL = 5.0


MAP_PROMPT_TEMPLATE = """以下の内容を重要な情報はできるだけ残して要約してください。


"{text}"


要約:"""

REDUCE_PROMPT_TEMPLATE = """以下の内容を全体を網羅して日本語で簡潔に要約してください。


"{text}"


簡潔な要約:"""

CONCISELY_PROMPT_TEMPLATE = \
"""この動画の内容を全体を網羅して{length}簡潔に要約してください。
この動画の内容は以下の通りです。

タイトル：
{title}

内容：
{content}

簡潔な要約：
"""
CONCISELY_PROMPT_TEMPLATE_VARIABLES = ["length", "title", "content"]

TOPIC_PROMPT_TEMPLATE = \
"""I am creating an agenda for Youtube videos.
Below are notes on creating an agenda, as well as video title and content.
Please follow the instructions carefully and create an agenda from the title and content.

Notes:
- Please create an agenda that covers the entire content of the video.
- Your agenda should include headings and some subheaddings for each heading.
- Create headings and subheadings that follow the flow of the story.
- Please include important keywords in the heading and subheading.
- Please include only one topic per heading or subheading.
- Please assign each heading a sequential number such as 1, 2, 3.
- Please keep each heading as concise as possible.
- Please add a "-" to the beginning of each subheading and output it as bullet points.
- Please keep each subheading as concise as possible.
- Please create the agenda in Japanese.

Title:
{title}

Content:
{content}

Agenda:
"""
TOPIC_PROMPT_TEMPLATE_VARIABLES = ["title", "content"]


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

        self.tmp_concise_summary: str = ""
        self.tmp_topic: List[TopicModel] = []


    def _debug (self, message: str, end: str = "\n", flush: bool = False) -> None:
        if self.debug is False:
            return
        print(message, end=end, flush=flush)
        return


    def run (self, mode:int = MODE_ALL) -> Optional[SummaryResultModel]:
        if self.loading is True:
            return self._run_with_loading(mode)
        return self._run(mode)


    def _run_with_loading (self, mode:int = MODE_CONCISE|MODE_DETAIL) -> Optional[SummaryResultModel]:
        summary: Optional[SummaryResultModel] = None
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_loading = executor.submit(self._loading)
            try:
                summary = self._run(mode)
            except Exception as e:
                raise e
            finally:
                self.loading_canceled = True
                while future_loading.done() is False:
                    time.sleep(0.5)
        return summary


    def _run (self, mode:int = MODE_ALL) -> SummaryResultModel:
        # mode調整
        if mode & (MODE_TOPIC | MODE_CONCISE) > 0:
            mode |= MODE_DETAIL

        # 準備
        self.chain = self._prepare_summarize_chain()
        self.chunks = self._prepare_transcriptions()

        # 詳細な要約
        detail_summary: List[str] = []
        if mode & MODE_DETAIL > 0:
            detail_summary = self._summarize_in_detail()

        # 以降は詳細な要約をベースに処理するので非同期で並行処理で行う
        (concise_summary, topic) = self._async_run_with_detail_summary(mode, detail_summary)

        summary: SummaryResultModel = SummaryResultModel(
            title=self.title,
            author=self.author,
            lengthSeconds=self.lengthSeconds,
            url=self.url,
            concise=concise_summary,
            detail=detail_summary,
            topic=topic,
        )

        # 全モードの場合のみ保存する
        if mode == MODE_ALL:
            if not os.path.isdir(os.path.dirname(self.summary_file)):
                os.makedirs(os.path.dirname(self.summary_file))
            with open(self.summary_file, "w") as f:
                f.write(summary.model_dump_json())

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
        MAXLENGTH = 3000
        OVERLAP_LENGTH = 10
        transcriptions: List[YoutubeTranscriptType] = YouTubeTranscriptApi.get_transcript(video_id=self.vid, languages=["ja", "en", "en-US"])
        return divide_transcriptions_into_chunks(
            transcriptions,
            maxlength = MAXLENGTH,
            overlap_length = OVERLAP_LENGTH,
            id_prefix = self.vid
        )


    def _summarize_in_detail (self) -> List[str]:
        if self.chain is None:
            return []
        chunk_groups: List[List[TranscriptChunkModel]] = self._divide_chunks_into_N_groups_evenly(5)
        tasks = [
            self.chain.arun([Document(page_content=chunk.text) for chunk in chunks]) for chunks in chunk_groups
        ]
        gather = asyncio.gather(*tasks)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(gather)


    def _async_run_with_detail_summary (self, mode, detail_summary) -> Tuple[str, List[TopicModel]]:
        if mode & (MODE_CONCISE | MODE_TOPIC) == 0:
            return ("", [])

        tasks = []
        if mode & MODE_CONCISE > 0:
            tasks.append(self._summarize_concisely(detail_summary))
        if mode & MODE_TOPIC > 0:
            tasks.append(self._extract_topic(detail_summary))

        gather = asyncio.gather(*tasks)
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(gather)

        concise_summary: str = ""
        topic: List[TopicModel] = []
        if mode & (MODE_CONCISE | MODE_TOPIC) == (MODE_CONCISE | MODE_TOPIC):
            concise_summary = results[0]
            topic = results[1]
        elif mode & MODE_CONCISE == MODE_CONCISE:
            concise_summary = results[0]
        elif mode & MODE_TOPIC == MODE_TOPIC:
            topic = results[0]

        return (concise_summary, topic)


    async def _summarize_concisely (self, detail_summary: List[str]) -> str:
        @retry(
            stop=stop_after_attempt(MAX_RETRY_COUNT),
            wait=wait_fixed(RETRY_INTERVAL),
        )
        async def _summarize () -> str:
            summary: str  = await chain.arun(**args)
            if len(summary) > MAX_CONCISE_SUMMARY_LENGTH * MAX_LENGTH_MARGIN_MULTIPLIER:
                if len(self.tmp_concise_summary) == 0:
                    self.tmp_concise_summary = summary
                elif len(summary) < len(self.tmp_concise_summary):
                    self.tmp_concise_summary = summary
                raise ValueError(f"summary too long. - ({len(summary)})")
            return summary

        # 準備
        prompt = PromptTemplate(
            template=CONCISELY_PROMPT_TEMPLATE,
            input_variables=CONCISELY_PROMPT_TEMPLATE_VARIABLES,
        )
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=self.debug,
        )
        length_str = ''
        if SPECIFY_SUMMARY_MAX_LENGTH is True:
            length_str = f'{MAX_CONCISE_SUMMARY_LENGTH}文字くらいで'
        args: Dict[str, str] = {
            "length": length_str,
            "title": self.title,
            "content": "\n".join(detail_summary),
        }
        self.tmp_concise_summary = ""
        concise_summary: str = ""

        # リトライしても改善されない場合は一番マシなもので我慢する
        try:
            concise_summary = await _summarize()
        except RetryError:
            concise_summary = self.tmp_concise_summary
        except Exception as e:
            raise e

        return concise_summary


    async def _extract_topic (self, detail_summary: List[str] = []) -> List[TopicModel]:
        @retry(
            stop=stop_after_attempt(MAX_RETRY_COUNT),
            wait=wait_fixed(RETRY_INTERVAL),
        )
        async def _extract () -> List[TopicModel]:
            result: str = await chain.arun(**args)
            topic: List[TopicModel] = self._parse_topic(result)
            if len(topic) > MAX_TOPIC_ITEMS:
                if len(self.tmp_topic) == 0:
                    self.tmp_topic = topic
                elif len(topic) < len(self.tmp_topic):
                    self.tmp_topic = topic
                raise ValueError(f"topic too much. - ({len(topic)})")
            return topic

        # 準備
        prompt = PromptTemplate(
            template=TOPIC_PROMPT_TEMPLATE,
            input_variables=TOPIC_PROMPT_TEMPLATE_VARIABLES,
        )
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=self.debug,
        )
        args: Dict[str, str] = {
            "title": self.title,
            "content": "\n".join(detail_summary) if len(detail_summary) > 0 else "",
        }
        self.tmp_topic = []
        topic: List[TopicModel] = []

        # リトライしても改善されない場合は一番マシなもので我慢する
        try:
            topic = await _extract()
        except RetryError:
            topic = self.tmp_topic
        except Exception as e:
            raise e

        return topic


    def _parse_topic (self, topic_string: str) -> List[TopicModel]:
        topics: List[TopicModel] = []
        topic: TopicModel = TopicModel(title="", abstract=[])
        for line in topic_string.split("\n"):
            line = line.strip()
            if line == "":
                continue
            if line[0].isdigit():
                if topic.title != "":
                    topics.append(topic)
                topic = TopicModel(title=line, abstract=[])
                continue
            if line[0] == "-":
                topic.abstract.append(line)
        return topics


    def _divide_chunks_into_N_groups_evenly (self, group_num: int = 5) -> List[List[TranscriptChunkModel]]:
        chunk_num: int = len(self.chunks)

        if chunk_num <= group_num:
            return [ [c] for c in self.chunks]

        deltas: List[int] = [(chunk_num // group_num) for _ in range(0, group_num)]
        extra: int = chunk_num % group_num
        for i in range(0, extra):
            deltas[i] += 1
        groups: List[List[TranscriptChunkModel]] = []
        idx: int = 0
        for delta in deltas:
            groups.append(self.chunks[idx: idx+delta])
            idx += delta

        return groups


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

    summary: Optional[SummaryResultModel] = None
    summary_file: str = f'{os.environ["SUMMARY_STORE_DIR"]}/{vid}'
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = SummaryResultModel(**(json.load(f)))
    else:
        # summary = YoutubeSummarize(vid, debug=True).run(mode=mode)
        summary = YoutubeSummarize(vid).run(mode=mode)

    if summary is None:
        return ""
    if mode == MODE_CONCISE:
        return summary.concise.strip()
    if mode == MODE_TOPIC:
        # return "\n".join(summary["topic"])
        return ""
    return "\n".join(summary.detail)
