from typing import List, Optional, Dict, Tuple, Any, TypedDict
import os
import sys
import json
import asyncio
import time
import re
import math
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
from .utils import setup_llm_from_environment, divide_transcriptions_into_chunks, count_tokens
from .types import SummaryResultModel, AgendaModel, DetailSummary, TopicModel


MODE_CONCISE  = 0x01
MODE_DETAIL   = 0x02
MODE_AGENDA   = 0x04
MODE_KEYWORD  = 0x08
MODE_TOPIC    = 0x10
MODE_ALL = MODE_CONCISE | MODE_DETAIL | MODE_AGENDA | MODE_KEYWORD | MODE_TOPIC

MAX_CONCISE_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH", "300"))
MAX_LENGTH_MARGIN_MULTIPLIER = float(os.getenv("MAX_SUMMARY_LENGTH_MARGIN", "2.0"))
MAX_AGENDA_ITEMS = int(os.getenv("MAX_AGENDA_ITEM", "10"))
MAX_AGENDA_ITEMS_MARGIN_MULTIPLIER = float(os.getenv("MAX_AGENDA_ITEM_MARGIN", "1.3"))
MAX_KEYWORDS = int(os.getenv("MAX_KEYWORD", "20"))
MAX_KEYWORDS_MARGIN_MULTIPLIER = float(os.getenv("MAX_KEYWORD_MARGIN", "1.3"))
EXCLUDE_KEYWORDS = [
    "セミナー", "企画", "チャンネル登録", "情報", "コラボ", "勉強会", "予定", "登録", "イベント",
    "メールマガジン",
]
MAX_TOPICS_BASE = int(os.getenv("MAX_TOPIC_BASE", "10"))
TOPICS_PER_HOUR = int(os.getenv("TOPIC_PER_HOUR", "5"))
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

AGENDA_PROMPT_TEMPLATE = \
"""I am creating an agenda for Youtube videos.
Below are notes on creating an agenda, as well as video content.
Please follow the notes carefully and create an agenda from the content.

Notes:
- Your agenda must cover the entire content.
- Your agenda must include headings and some subheaddings for each heading.
- Subheadings on similar subjects should be included under one heading.
- Please keep each heading and subheading short and concise term, not sentence.
- Please assign each heading a sequential number such as 1, 2, 3.
- Do not assign subheadings numbers such as 1.1, 3.1 to, but instead output them as bulleted lists with a "-" instead.
- Please include no more than {max} headings.
- Please create the agenda in Japanese.

Content:
{content}

Agenda:
"""
AGENDA_PROMPT_TEMPLATE_VARIABLES = ["content", "max"]

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

TOPIC_PROMPT_TEMPLATE = """\
あなたは優秀な編集者です。
ある動画から視聴者の興味を引くトピックを抽出することを仕事としています。
次の手順でトピックを抽出し、トピックとその開始時刻を指定されたフォーマットで出力してください。
トピックを抽出するときは以下の条件を守ってください。

## 手順
1. 「## 概要」に続く内容を参照してトピックを抽出する
2. 「## 文字起こし」に続く内容を参照してトピックの開始時刻を求める
3. 開始時刻とトピックを出力する

## トピック抽出の条件
- トピックは最大で{max}個までとする。
- トピックは30文字以内で見出しとなるように分かりやすく簡潔に要約すること。
- 関連するトピックが連続する場合は1つにまとめること。
- トピックは日本語で出力すること。

## 出力のフォーマット
1. [開始時刻1] トピック1
2. [開始時刻2] トピック2
3. [開始時刻3] トピック3

## 概要
{abstract}

## 文字起こし
{transcript}

## トピック（日本語で）
1. [
"""

TOPIC_PROMPT_TEMPLATE_VARIABLES = ["max", "abstract", "transcript"]


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
        self.transcriptions: List[YoutubeTranscriptType] = []

        self.llm: LLMType = setup_llm_from_environment()

        self.loading: bool = loading
        self.loading_canceled: bool = False

        self.tmp_concise_summary: str = ""
        self.tmp_agenda: List[AgendaModel] = []
        self.tmp_keyword: List[str] = []
        self.tmp_topic: List[TopicModel] = []


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
        if mode & MODE_AGENDA > 0:
            print("[Agenda]")
            for agenda in summary.agenda:
                print(f'{agenda.title}')
                if len(agenda.subtitle) > 0:
                    print("  ", "\n  ".join(agenda.subtitle), sep="")
            print("")
        if mode & MODE_TOPIC > 0:
            print("[Topic]")
            for topic in summary.topic:
                print(f'- {topic.topic} [{topic.time}]')
            print("")
        if mode & MODE_DETAIL > 0:
            detail_summary: List[str] = [ s.text for s in summary.detail]
            print("[Detail Summary]"); print("・", "\n・".join(detail_summary)); print("")
        return


    @classmethod
    def summary (cls, vid: Optional[str], refresh: bool = False) -> Optional[SummaryResultModel]:
        if vid is None:
            return None
        summary: Optional[SummaryResultModel] = None
        summary_file: str = f'{os.environ["SUMMARY_STORE_DIR"]}/{vid}'
        if refresh is False and os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = SummaryResultModel(**(json.load(f)))
        else:
            # summary = cls(vid, debug=True).run()
            summary = cls(vid).run()
        return summary


    @classmethod
    async def asummary (cls, vid: Optional[str], refresh: bool = False) -> Optional[SummaryResultModel]:
        if vid is None:
            return None
        summary: Optional[SummaryResultModel] = None
        summary_file: str = f'{os.environ["SUMMARY_STORE_DIR"]}/{vid}'
        if refresh is False and os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = SummaryResultModel(**(json.load(f)))
        else:
            summary = await cls(vid).arun()
        return summary


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


    async def _arun_with_loading (self, mode:int = MODE_ALL) -> Optional[SummaryResultModel]:
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

        # 字幕を取得
        self.transcriptions = YouTubeTranscriptApi.get_transcript(
            video_id=self.vid, languages=["ja", "en", "en-US"]
        )

        # 詳細な要約
        detail_summary: List[DetailSummary] = await self._summarize_in_detail()

        # 簡潔な要約、トピック抽出、キーワード抽出（非同期で並行処理）
        (concise_summary, agenda, keyword, topic) = await self._arun_with_detail_summary(mode, detail_summary)

        summary: SummaryResultModel = SummaryResultModel(
            title=self.title,
            author=self.author,
            lengthSeconds=self.lengthSeconds,
            url=self.url,
            concise=concise_summary,
            detail=detail_summary,
            agenda=agenda,
            keyword=keyword,
            topic=topic,
        )

        # 全モードの場合のみ保存する
        if mode == MODE_ALL:
            if not os.path.isdir(os.path.dirname(self.summary_file)):
                os.makedirs(os.path.dirname(self.summary_file))
            with open(self.summary_file, "w") as f:
                f.write(summary.model_dump_json())

        return summary


    async def _summarize_in_detail (self) -> List[DetailSummary]:
        def _prepare_summarize_chain () -> BaseCombineDocumentsChain:
            return load_summarize_chain(
                llm=self.llm,
                chain_type='map_reduce',
                map_prompt=PromptTemplate(template=MAP_PROMPT_TEMPLATE, input_variables=["text"]),
                combine_prompt=PromptTemplate(template=REDUCE_PROMPT_TEMPLATE, input_variables=["text"]),
                verbose=self.debug
            )

        def _prepare_transcriptions (
                maxlength: int = 3000, minlength: int = 500, step_length: int = 500,
                overlap_length: int = 10, step_overlap: int = 2,
                min_chunks: int = 5,
        ) -> List[TranscriptChunkModel]:
            chunks: List[TranscriptChunkModel] | None = None
            while True:
                chunks = divide_transcriptions_into_chunks(
                    self.transcriptions,
                    maxlength = maxlength,
                    overlap_length = overlap_length,
                    id_prefix = self.vid
                )
                if chunks is not None and len(chunks) >= min_chunks:
                    break
                maxlength -= step_length
                overlap_length -= step_overlap
                if maxlength < minlength or overlap_length < 1:
                    break
            return chunks

        class SummaryChainResultType (TypedDict):
            input_documents: List[Document]
            output_text: str


        chain: BaseCombineDocumentsChain = _prepare_summarize_chain()
        chunks: List[TranscriptChunkModel] = _prepare_transcriptions()

        chunk_groups: List[List[TranscriptChunkModel]] = self._divide_chunks_into_N_groups(chunks, 5, 12000)
        tasks = [
            chain.ainvoke(
                input={"input_documents": [Document(page_content=chunk.text) for chunk in chunks]}
            ) for chunks in chunk_groups
        ]
        summaries: List[SummaryChainResultType] = await asyncio.gather(*tasks)
        results: List[DetailSummary] = [
            DetailSummary(text=s["output_text"], start=c[0].start) for c, s in zip(chunk_groups, summaries)
        ]

        return results


    async def _arun_with_detail_summary (
        self,
        mode: int,
        detail_summary: List[DetailSummary]
    ) -> Tuple[str, List[AgendaModel], List[str], List[TopicModel]]:
        if mode & (MODE_CONCISE | MODE_AGENDA | MODE_KEYWORD | MODE_TOPIC) == 0:
            return ("", [], [], [])

        summaries: List[str] = [ s.text for s in detail_summary]
        tasks = []
        tasks.append(self._summarize_concisely(summaries, mode & MODE_CONCISE > 0))
        tasks.append(self._extract_keyword(summaries, mode & MODE_KEYWORD > 0))
        tasks.append(self._extract_agenda(summaries, mode & MODE_AGENDA > 0))
        tasks.append(self._extract_topic(detail_summary, mode & MODE_TOPIC > 0))

        results: List[Any] = await asyncio.gather(*tasks)

        concise_summary: str      = results[0]
        keyword: List[str]        = results[1]
        agenda: List[AgendaModel] = results[2]
        topic: List[TopicModel]   = results[3]

        return (concise_summary, agenda, keyword, topic)


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
            summary: str = (await chain.ainvoke(input=args))["text"]
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
            result: str = (await chain.ainvoke(input=args))["text"]
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


    async def _extract_agenda (
        self,
        detail_summary: List[str],
        enable: bool = True
    ) -> List[AgendaModel]:
        def _check (agenda: List[AgendaModel]) -> bool:
            if len(agenda) > MAX_AGENDA_ITEMS * MAX_AGENDA_ITEMS_MARGIN_MULTIPLIER:
                if len(self.tmp_agenda) == 0:
                    self.tmp_agenda = agenda
                elif len(agenda) < len(self.tmp_agenda):
                    self.tmp_agenda = agenda
                self._debug(f"agenda too much. - ({len(agenda)})")
                return False
            if len(agenda) == 0:
                self._debug(f"agenda is empty.\n{agenda}")
                return False
            return True

        def _parse_agenda (agenda_string: str) -> List[AgendaModel]:
            agenda: List[AgendaModel] = []
            one_agenda: Optional[AgendaModel] = None
            for line in agenda_string.split("\n"):
                line = line.strip()
                if line == "":
                    continue
                if line[0].isdigit():
                    if one_agenda is not None:
                        agenda.append(one_agenda)
                    one_agenda = AgendaModel(title=line, subtitle=[], time=[])
                    continue
                if one_agenda is not None:
                    one_agenda.subtitle.append(line if line[0] == "-" else f"- {line}")
            if one_agenda is not None:
                agenda.append(one_agenda)
            return agenda

        @retry(
            stop=stop_after_attempt(MAX_RETRY_COUNT),
            wait=wait_fixed(RETRY_INTERVAL),
            retry=retry_if_not_result(_check)
        )
        async def _extract () -> List[AgendaModel]:
            result: str = (await chain.ainvoke(input=args))["text"]
            agenda: List[AgendaModel] = _parse_agenda(result)
            return agenda


        if enable is False:
            return []

        # 準備
        prompt = PromptTemplate(
            template=AGENDA_PROMPT_TEMPLATE,
            input_variables=AGENDA_PROMPT_TEMPLATE_VARIABLES,
        )
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=self.debug,
        )
        args: Dict[str, str|int] = {
            "content": "\n".join(detail_summary) if len(detail_summary) > 0 else "",
            "max": MAX_AGENDA_ITEMS,
        }
        self.tmp_agenda = []
        agenda: List[AgendaModel] = []

        # リトライしても改善されない場合は一番マシなもので我慢する
        try:
            agenda = await _extract()
        except RetryError:
            agenda = self.tmp_agenda
        except Exception as e:
            raise e

        return agenda


    async def _extract_topic (
        self,
        detail_summary: List[DetailSummary] = [],
        enable: bool = True,
    ) -> List[TopicModel]:
        class TopicArgType (TypedDict):
            max: int
            abstract: str
            transcript: str

        def _s2hms (seconds: float) -> str:
            seconds = math.floor(seconds)
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            return "%d:%02d:%02d" % (h, m, s)

        def _get_max_topics () -> int:
            hour = self.lengthSeconds / 3600
            return MAX_TOPICS_BASE + int((hour - 1) * TOPICS_PER_HOUR)

        def _to_int_with_round (value: float) -> int:
                int_val = int(value)
                diff = value - int_val
                if diff >= 0.5:
                    int_val += 1
                return int_val

        @retry(
            stop=stop_after_attempt(MAX_RETRY_COUNT),
            wait=wait_fixed(RETRY_INTERVAL),
        )
        async def _topic (args: TopicArgType) -> List[TopicModel]:
            result: str = (await chain.ainvoke(input=args))["text"] # type: ignore
            if not re.match(r"^1. \[", result):
                result = "1. [" + result
            topics: List[TopicModel] = []
            for result in result.split("\n"):
                if result.strip() == "":
                    continue
                if result.startswith("#") is True:
                    break
                try:
                    _, time, topic = re.split(r"\s+", result, maxsplit=2)
                except:
                    continue
                time = re.sub(r"[\[\]]", "", time)
                topics.append(TopicModel(topic=topic, time=time))
            return topics

        def _mk_args (max: int, abstract: str, transcript: str) -> TopicArgType:
            return {
                "max": max,
                "abstract": abstract,
                "transcript": transcript,
            }


        if enable is False:
            return []

        chunks: List[TranscriptChunkModel] = divide_transcriptions_into_chunks(
            self.transcriptions,
            maxlength = 100,
            overlap_length = 0,
            id_prefix = self.vid
        )
        contents: List[str] = []
        groups: List[List[str]] = []
        s_idx: int = 0
        s_endtime: float = detail_summary[s_idx + 1].start if s_idx + 1 < len(detail_summary) else self.lengthSeconds
        for chunk in chunks:
            start_time: str = _s2hms(chunk.start)
            transcript: str = f'[{start_time}] {chunk.text}'.replace("\n", " ")
            if chunk.start >= s_endtime:
                groups.append(contents)
                contents = []
                s_idx += 1
                s_endtime = detail_summary[s_idx + 1].start if s_idx + 1 < len(detail_summary) else self.lengthSeconds
            contents.append(transcript)
        if len(contents) > 0:
            groups.append(contents)

        max_topics = _get_max_topics()
        tokens: int = 0
        transcript: str = ""
        abstract: str = ""
        max: int = 0
        args_list: List[TopicArgType] = []
        for group, summary in zip(groups, detail_summary):
            tmp_transcript: str = "\n\n".join(group)
            add_tokens: int = count_tokens(summary.text) + count_tokens(tmp_transcript)
            if tokens > 0 and tokens + add_tokens > 14000:
                args_list.append(_mk_args(max, abstract, transcript))
                transcript = ""; abstract = ""; max = 0; tokens = 0 # 初期化
            max += _to_int_with_round(max_topics * (len(group)/len(chunks)))
            transcript += "\n\n" + tmp_transcript
            abstract += "\n" + summary.text
            tokens += count_tokens(summary.text) + count_tokens(tmp_transcript)
        if transcript != "" and abstract != "":
            args_list.append(_mk_args(max, abstract, transcript))

        prompt = PromptTemplate(
            template=TOPIC_PROMPT_TEMPLATE,
            input_variables=TOPIC_PROMPT_TEMPLATE_VARIABLES,
        )
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=self.debug,
        )
        tasks = [ _topic(args) for args in args_list ]

        try:
            results: List[Any] = await asyncio.gather(*tasks)
            topics: List[TopicModel] = []
            for result in results:
                topics.extend(result)
        except Exception as e:
            raise e

        return topics


    def _divide_chunks_into_N_groups (
            self,
            chunks: List[TranscriptChunkModel] = [],
            group_num: int = 5,
            max_tokens: int = 12000,
    ) -> List[List[TranscriptChunkModel]]:
        groups: List[List[TranscriptChunkModel]] = []
        while True:
            is_ok: bool = True
            groups = self._divide_chunks_into_N_groups_evenly(chunks, group_num)
            for group in groups:
                tokens: int = 0
                for chunk in group:
                    tokens += count_tokens(chunk.text)
                if tokens > max_tokens:
                    is_ok = False
                    break
            if is_ok:
                break
            group_num += 1
        return groups


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


    def _divide_chunks_into_N_groups_1 (
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

