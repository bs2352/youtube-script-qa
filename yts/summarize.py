from typing import List, Optional, Dict, Tuple, Any
import os
import sys
import json
import asyncio
import time
import re
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError, retry_if_not_result

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
MODE_KEYWORD = 0x08
MODE_ALL     = MODE_CONCISE | MODE_DETAIL | MODE_TOPIC | MODE_KEYWORD

MAX_CONCISE_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH", "300"))
MAX_LENGTH_MARGIN_MULTIPLIER = float(os.getenv("MAX_SUMMARY_LENGTH_MARGIN", "2.0"))
MAX_TOPIC_ITEMS = int(os.getenv("MAX_TOPIC_ITEM", "10"))
MAX_TOPIC_ITEMS_MARGIN_MULTIPLIER = float(os.getenv("MAX_TOPIC_ITEM_MARGIN", "1.3"))
MAX_KEYWORDS = int(os.getenv("MAX_KEYWORD", "20"))
MAX_KEYWORDS_MARGIN_MULTIPLIER = float(os.getenv("MAX_KEYWORD_MARGIN", "1.3"))
EXCLUDE_KEYWORDS = [
    "セミナー", "企画", "チャンネル登録", "情報", "コラボ", "勉強会", "予定", "登録", "イベント",
]
MAX_RETRY_COUNT = int(os.getenv("MAX_RETRY_COUNT", "3"))
RETRY_INTERVAL = 5.0


MAP_PROMPT_TEMPLATE = """以下の内容を重要な情報はできるだけ残して要約してください。


"{text}"


要約:"""

REDUCE_PROMPT_TEMPLATE = """以下の内容を全体を網羅して日本語で簡潔に要約してください。


"{text}"


簡潔な要約:"""

CONCISELY_PROMPT_TEMPLATE = \
"""以下の内容をターゲットを絞って日本語で簡潔に要約してください。
要約は{max}文字以内にしてください。

内容：
{content}

簡潔な要約：
"""
CONCISELY_PROMPT_TEMPLATE_VARIABLES = ["content", "max"]

TOPIC_PROMPT_TEMPLATE = \
"""I am creating an agenda for Youtube videos.
Below are notes on creating an agenda, as well as video content.
Please follow the notes carefully and create an agenda from the content.

Notes:
- Your agenda must cover the entire content.
- Your agenda must include headings and some subheaddings for each heading.
- Please assign each heading a sequential number such as 1, 2, 3.
- Do not assign subheadings numbers such as 1.1, 3.1 to, but instead output them as bulleted lists with a "-" instead.
- Please include no more than {max} headings.
- Please keep each heading and subheading short and concise term, not sentence.
- Please create the agenda in Japanese.

Content:
{content}

Agenda:
"""
TOPIC_PROMPT_TEMPLATE_VARIABLES = ["content", "max"]

KEYWORD_PROMPT_TEMPLATE = \
"""Please extract impressive keywords from the video content listed below.
Please follow the notes carefully when extracting keywords.

Notes:
- Please scan the entire content first, then extract only targeted keywords.
- Please assign each keyword a sequential number such as 1, 2, 3.
- Please extract no more than {max} keywords.
- Do not output same keywords.
- Do not output similar keywords.
- Do not translate keywords into English.
- Please output one keyword in one line.

Content:
{content}

Keywords:
"""
KEYWORD_PROMPT_TEMPLATE_VARIABLES = ["content", "max"]


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

        self.url: str = f'https://www.youtube.com/watch?v={vid}'
        video_info = YouTube(self.url).vid_info["videoDetails"]
        self.title: str = video_info['title']
        self.author: str = video_info['author']
        self.lengthSeconds: int = int(video_info['lengthSeconds'])

        self.llm: LLMType = setup_llm_from_environment()

        self.loading: bool = loading
        self.loading_canceled: bool = False

        self.tmp_concise_summary: str = ""
        self.tmp_topic: List[TopicModel] = []
        self.tmp_keyword: List[str] = []


    def _debug (self, message: str, end: str = "\n", flush: bool = False) -> None:
        if self.debug is False:
            return
        print(message, end=end, flush=flush)
        return


    @classmethod
    def print (
        cls,
        summary: Optional[SummaryResultModel] = None,
        mode: int = MODE_ALL
    ) -> None:
        if summary is None:
            return
        print("[Title]"); print(summary.title); print("")
        if mode & MODE_CONCISE > 0:
            print("[Summary]"); print(summary.concise); print("")
        if mode & MODE_KEYWORD > 0:
            print("[Keyword]"); print(", ".join(summary.keyword)); print("")
        if mode & MODE_TOPIC > 0:
            print("[Topic]")
            for topic in summary.topic:
                print(f'{topic.title}')
                if len(topic.abstract) > 0:
                    print("  ", "\n  ".join(topic.abstract), sep="")
            print("")
        if mode & MODE_DETAIL > 0:
            print("[Detail Summary]"); print("・", "\n・".join(summary.detail)); print("")
        return


    def run (self, mode:int = MODE_ALL) -> Optional[SummaryResultModel]:
        loop = asyncio.get_event_loop()
        tasks = [self.arun(mode)]
        gather = asyncio.gather(*tasks)
        result = loop.run_until_complete(gather)[0]
        return result


    async def arun (self, mode:int = MODE_ALL) -> Optional[SummaryResultModel]:
        if self.loading is True:
            return await self._arun_with_loading(mode)
        return await self._arun(mode)


    async def _arun_with_loading (self, mode:int = MODE_CONCISE|MODE_DETAIL) -> Optional[SummaryResultModel]:
        summary: Optional[SummaryResultModel] = None
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_loading = executor.submit(self._loading)
            try:
                summary = await self._arun(mode)
            except Exception as e:
                raise e
            finally:
                self.loading_canceled = True
                while future_loading.done() is False:
                    time.sleep(0.5)
        return summary


    async def _arun (self, mode:int = MODE_ALL) -> SummaryResultModel:
        # mode調整
        if mode & MODE_DETAIL == 0:
            mode |= MODE_DETAIL

        # 詳細な要約
        detail_summary: List[str] = await self._summarize_in_detail()

        # 簡潔な要約、トピック抽出、キーワード抽出（非同期で並行処理）
        (concise_summary, topic, keyword) = await self._arun_with_detail_summary(mode, detail_summary)

        summary: SummaryResultModel = SummaryResultModel(
            title=self.title,
            author=self.author,
            lengthSeconds=self.lengthSeconds,
            url=self.url,
            concise=concise_summary,
            detail=detail_summary,
            topic=topic,
            keyword=keyword
        )

        # 全モードの場合のみ保存する
        if mode == MODE_ALL:
            if not os.path.isdir(os.path.dirname(self.summary_file)):
                os.makedirs(os.path.dirname(self.summary_file))
            with open(self.summary_file, "w") as f:
                f.write(summary.model_dump_json())

        return summary


    async def _summarize_in_detail (self) -> List[str]:
        def _prepare_summarize_chain () -> BaseCombineDocumentsChain:
            return load_summarize_chain(
                llm=self.llm,
                chain_type='map_reduce',
                map_prompt=PromptTemplate(template=MAP_PROMPT_TEMPLATE, input_variables=["text"]),
                combine_prompt=PromptTemplate(template=REDUCE_PROMPT_TEMPLATE, input_variables=["text"]),
                verbose=self.debug
            )

        def _prepare_transcriptions () -> List[TranscriptChunkModel]:
            MAXLENGTH = 3000
            OVERLAP_LENGTH = 10
            transcriptions: List[YoutubeTranscriptType] = YouTubeTranscriptApi.get_transcript(
                video_id=self.vid, languages=["ja", "en", "en-US"]
            )
            return divide_transcriptions_into_chunks(
                transcriptions,
                maxlength = MAXLENGTH,
                overlap_length = OVERLAP_LENGTH,
                id_prefix = self.vid
            )

        chain: BaseCombineDocumentsChain = _prepare_summarize_chain()
        chunks: List[TranscriptChunkModel] = _prepare_transcriptions()

        chunk_groups: List[List[TranscriptChunkModel]] = self._divide_chunks_into_N_groups_evenly(chunks, 5)
        tasks = [
            chain.arun([Document(page_content=chunk.text) for chunk in chunks]) for chunks in chunk_groups
        ]
        results: List[str] = await asyncio.gather(*tasks)

        return results


    async def _arun_with_detail_summary (
        self,
        mode,
        detail_summary
    ) -> Tuple[str, List[TopicModel], List[str]]:
        if mode & (MODE_CONCISE | MODE_TOPIC | MODE_KEYWORD) == 0:
            return ("", [], [])

        tasks = []
        tasks.append(self._summarize_concisely(detail_summary, mode & MODE_CONCISE > 0))
        tasks.append(self._extract_keyword(detail_summary, mode & MODE_KEYWORD > 0))
        tasks.append(self._extract_topic(detail_summary, mode & MODE_TOPIC > 0))

        results: List[Any] = await asyncio.gather(*tasks)

        concise_summary: str    = results[0]
        keyword: List[str]      = results[1]
        topic: List[TopicModel] = results[2]

        return (concise_summary, topic, keyword)


    async def _summarize_concisely (
        self,
        detail_summary: List[str],
        enable: bool = True
    ) -> str:
        def _check (summary:str) -> bool:
            if len(summary) > MAX_CONCISE_SUMMARY_LENGTH * MAX_LENGTH_MARGIN_MULTIPLIER:
                if len(self.tmp_concise_summary) == 0:
                    self.tmp_concise_summary = summary
                elif len(summary) < len(self.tmp_concise_summary):
                    self.tmp_concise_summary = summary
                self._debug(f"summary too long. - ({len(summary)})")
                return False
            return True

        @retry(
            stop=stop_after_attempt(MAX_RETRY_COUNT),
            wait=wait_fixed(RETRY_INTERVAL),
            retry=retry_if_not_result(_check)
        )
        async def _summarize () -> str:
            summary: str  = await chain.arun(**args)
            return summary


        if enable is False:
            return ""

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
        args: Dict[str, str|int] = {
            "content": "\n".join(detail_summary),
            "max": MAX_CONCISE_SUMMARY_LENGTH,
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


    async def _extract_keyword (
        self,
        detail_summary: List[str],
        enable: bool = True
    ) -> List[str]:
        def _check (keyword: List[str]) -> bool:
            if len(keyword) > MAX_KEYWORDS * MAX_KEYWORDS_MARGIN_MULTIPLIER:
                if len(self.tmp_keyword) == 0:
                    self.tmp_keyword = keyword
                elif len(keyword) < len(self.tmp_keyword):
                    self.tmp_keyword = keyword
                self._debug(f"keyword too much. - ({len(keyword)})")
                return False
            if len(keyword) == 1:
                self._debug(f"Perhaps invalid format. \n{keyword[0]}")
                return False
            return True

        def _trim_kwd (kwd: str) -> str:
            keyword: str = re.sub(r"^\d+\.?", "", kwd.strip()).strip()
            return keyword

        @retry(
            stop=stop_after_attempt(MAX_RETRY_COUNT),
            wait=wait_fixed(RETRY_INTERVAL),
            retry=retry_if_not_result(_check)
        )
        async def _extract () -> List[str]:
            result: str = await chain.arun(**args)
            keyword: List[str] = [ _trim_kwd(kwd) for kwd in result.split("\n")]
            keyword = [kwd for kwd in keyword if kwd not in EXCLUDE_KEYWORDS ]
            return keyword


        if enable is False:
            return []

        # 準備
        prompt = PromptTemplate(
            template=KEYWORD_PROMPT_TEMPLATE,
            input_variables=KEYWORD_PROMPT_TEMPLATE_VARIABLES,
        )
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=self.debug,
        )
        args: Dict[str, str|int] = {
            "content": "\n".join(detail_summary) if len(detail_summary) > 0 else "",
            "max": MAX_KEYWORDS,
        }
        self.tmp_keyword = []
        keyword: List[str] = []

        # リトライしても改善されない場合は一番マシなもので我慢する
        try:
            keyword = await _extract()
        except RetryError:
            keyword = self.tmp_keyword
        except Exception as e:
            raise e

        return keyword


    async def _extract_topic (
        self,
        detail_summary: List[str],
        enable: bool = True
    ) -> List[TopicModel]:
        def _check (topic: List[TopicModel]) -> bool:
            if len(topic) > MAX_TOPIC_ITEMS * MAX_TOPIC_ITEMS_MARGIN_MULTIPLIER:
                if len(self.tmp_topic) == 0:
                    self.tmp_topic = topic
                elif len(topic) < len(self.tmp_topic):
                    self.tmp_topic = topic
                self._debug(f"topic too much. - ({len(topic)})")
                return False
            if len(topic) == 0:
                self._debug(f"topic is empty.\n{topic}")
                return False
            return True

        def _parse_topic (topic_string: str) -> List[TopicModel]:
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

        @retry(
            stop=stop_after_attempt(MAX_RETRY_COUNT),
            wait=wait_fixed(RETRY_INTERVAL),
            retry=retry_if_not_result(_check)
        )
        async def _extract () -> List[TopicModel]:
            result: str = await chain.arun(**args)
            topic: List[TopicModel] = _parse_topic(result)
            return topic


        if enable is False:
            return []

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
        args: Dict[str, str|int] = {
            "content": "\n".join(detail_summary) if len(detail_summary) > 0 else "",
            "max": MAX_TOPIC_ITEMS,
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


    def _divide_chunks_into_N_groups_evenly (
        self,
        chunks: List[TranscriptChunkModel] = [],
        group_num: int = 5
    ) -> List[List[TranscriptChunkModel]]:

        chunk_num: int = len(chunks)

        if chunk_num <= group_num:
            return [ [c] for c in chunks]

        deltas: List[int] = [(chunk_num // group_num) for _ in range(0, group_num)]
        extra: int = chunk_num % group_num
        for i in range(0, extra):
            deltas[i] += 1

        groups: List[List[TranscriptChunkModel]] = []
        idx: int = 0
        for delta in deltas:
            groups.append(chunks[idx: idx+delta])
            idx += delta

        return groups


    def _divide_chunks_into_N_groups (
        self,
        chunks: List[TranscriptChunkModel] = [],
        group_num: int = 5
    ) -> List[List[TranscriptChunkModel]]:

        total_time: float = chunks[-1].start + chunks[-1].duration
        delta: float = total_time // group_num
        groups: List[List[TranscriptChunkModel]] = [[] for _ in range(0, group_num)]
        for chunk in chunks:
            idx = int(chunk.start // delta)
            idx = min(idx, group_num - 1)
            groups[idx].append(chunk)

        return [group  for group in groups if len(group) > 0]


    def _divide_chunks_into_groups_by_time_interval (
        self,
        chunks: List[TranscriptChunkModel] = [],
        interval_minutes: int = 5
    ) -> List[List[TranscriptChunkModel]]:

        total_time: float = chunks[-1].start + chunks[-1].duration
        delta: int = interval_minutes * 60
        group_num: int = (int(total_time) + 1) // delta
        groups: List[List[TranscriptChunkModel]] = [[] for _ in range(0, group_num)]
        for chunk in chunks:
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


def get_summary (vid: str, mode: int = MODE_ALL) -> Optional[SummaryResultModel]:
    if mode & MODE_ALL == 0:
        raise ValueError("mode is invalid.")

    summary: Optional[SummaryResultModel] = None
    summary_file: str = f'{os.environ["SUMMARY_STORE_DIR"]}/{vid}'
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = SummaryResultModel(**(json.load(f)))
    else:
        # summary = YoutubeSummarize(vid, debug=True).run(mode=mode)
        summary = YoutubeSummarize(vid).run(mode=mode)
    return summary
