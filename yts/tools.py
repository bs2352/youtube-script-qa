from typing import Tuple, List, Any, Optional
import re
import numpy
import math
import asyncio
import os

from .types import SummaryResultModel
from .summarize import YoutubeSummarize
from .qa import YoutubeQA
from .utils import setup_embedding_from_environment

AGENDA_TIME_TABLE_RETRIEVE_NUM = int(os.getenv("AGENDA_TIME_TABLE_RETRIEVE_NUM", "3"))
TOPIC_TIME_TABLE_RETRIEVE_NUM = int(os.getenv("TOPIC_TIME_TABLE_RETRIEVE_NUM", "5"))

class YoutubeAgendaTimeTable:
    @classmethod
    async def amake (
        cls,
        vid: str = "",
        summary: Optional[SummaryResultModel] = None,
        store: bool = False,
    ) -> SummaryResultModel:

        def _cosine_similarity(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
            return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

        def _s2hms (seconds: int) -> str:
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            return "%d:%02d:%02d" % (h, m, s)

        def _hms2s (hms: str) -> int:
            h, m, s = hms.split(':')
            return int(h) * 3600 + int(m) * 60 + int(s)

        async def _aget_likely_summary (summary: SummaryResultModel) -> Tuple[List[int], List[numpy.ndarray]]:
            def __get_agenda_items (summary: SummaryResultModel) -> List[str]:
                items: List[str] = []
                for agenda in summary.agenda:
                    title: str = re.sub(r"^\d+\.?", "", agenda.title).strip()
                    if len(agenda.subtitle) == 0:
                        items.append(title)
                        continue
                    for subtitle in agenda.subtitle:
                        items.append(f"{title} {subtitle}")
                return items

            llm_embedding = setup_embedding_from_environment()
            tasks = [
                llm_embedding.aembed_documents([ d.text for d in summary.detail]),
                llm_embedding.aembed_documents(__get_agenda_items(summary))
            ]
            results: List[Any] = await asyncio.gather(*tasks)
            summary_embs: numpy.ndarray = numpy.array(results[0])
            agenda_embs: List[numpy.ndarray] = [ numpy.array(a) for a in results[1]]

            idx: int = 0
            likely_summary: List[int] = []
            similarities_list: List[numpy.ndarray] = []
            for agenda in summary.agenda:
                if len(agenda.subtitle) == 0:
                    similarities = numpy.array([
                        _cosine_similarity(agenda_embs[idx], summary_emb) for summary_emb in summary_embs
                    ])
                    index = int(numpy.argmax(similarities))
                    likely_summary.append(index)
                    similarities_list.append(similarities)
                    idx += 1
                    continue
                for _ in agenda.subtitle:
                    similarities = numpy.array([
                        _cosine_similarity(agenda_embs[idx], summary_emb) for summary_emb in summary_embs
                    ])
                    index = int(numpy.argmax(similarities))
                    likely_summary.append(index)
                    similarities_list.append(similarities)
                    idx += 1

            return (likely_summary, similarities_list)

        def _fix_likely_summary(
            likely_summary: List[int], similarities_list: List[numpy.ndarray]
        ) -> List[int]:
            def __which_is_most_similar (
                candidates_idx: List[int], similarities: numpy.ndarray
            ) -> int:
                similar_idx: int = candidates_idx[0]
                for idx in candidates_idx:
                    if idx >= len(similarities):
                        break
                    if similarities[similar_idx] < similarities[idx]:
                        similar_idx = idx
                return similar_idx

            def __force_change (
                likely_summary: List[int], similarities_list: List[numpy.ndarray]
            ) -> List[int]:
                if len(likely_summary) <= 2:
                    return likely_summary
                if len(likely_summary) < len(similarities_list[0]) + 2:
                    return likely_summary
                likely_summary[0] = 0
                likely_summary[-1] = len(similarities_list[0]) - 1
                return likely_summary


            def __compare_with_neiborhood (
                likely_summary: List[int], similarities_list: List[numpy.ndarray],
            ) -> List[int]:
                def ___get_next_summary (base_index: int, likely_summary: List[int]) -> int:
                    for idx in range(base_index+1, len(likely_summary)):
                        if likely_summary[idx] != likely_summary[base_index]:
                            return likely_summary[idx]
                    return likely_summary[base_index]

                fixed: List[int] = []
                for idx, cur_summary in enumerate(likely_summary):
                    if idx == 0 or idx == len(likely_summary) - 1:
                        fixed.append(cur_summary)
                        continue
                    similarities: numpy.ndarray = similarities_list[idx]
                    prev_summary = fixed[idx-1] if idx > 0 else 0
                    next_summary = ___get_next_summary(idx, likely_summary)
                    if prev_summary > cur_summary:
                        if prev_summary >= next_summary:
                            fixed.append(prev_summary)
                        else:
                            candidates: List[int] = [ i for i in range(prev_summary, next_summary)]
                            fixed.append(__which_is_most_similar(candidates, similarities))
                        continue
                    if cur_summary - prev_summary > 2:
                        candidates: List[int] = [prev_summary, prev_summary + 1]
                        fixed.append(__which_is_most_similar(candidates, similarities))
                        continue
                    if cur_summary - prev_summary == 2:
                        fixed.append(prev_summary + 1)
                        continue
                    candidates: List[int] = [prev_summary, prev_summary + 1]
                    fixed.append(__which_is_most_similar(candidates, similarities))

                return fixed

            def __check_sequence (likely_summary: List[int]) -> bool:
                for idx in range(len(likely_summary)):
                    cur = likely_summary[idx]
                    prev = likely_summary[idx-1] if idx > 0 else likely_summary[0]
                    next = likely_summary[idx+1] if idx + 1 < len(likely_summary) else likely_summary[-1]
                    if cur < prev or cur > next:
                        return False
                return True

            def __fix (
                likely_summary: List[int], similarities_list: List[numpy.ndarray]
            ) -> List[int]:
                likely_summary = __force_change(likely_summary, similarities_list)
                likely_summary = __compare_with_neiborhood(likely_summary, similarities_list)
                return likely_summary

            for _ in range(5):
                likely_summary = __fix(likely_summary, similarities_list)
                if __check_sequence(likely_summary):
                    break

            return likely_summary

        def _get_time_range (index: int, summary: SummaryResultModel) -> Tuple[str, str]:
            start = _s2hms(math.floor(summary.detail[index].start))
            end = _s2hms(summary.lengthSeconds)
            if index + 1 < len(summary.detail):
                end = _s2hms(math.floor(summary.detail[index + 1].start))
            return (start, end)

        def _mk_summary_priority (s_index: int, simimalities: numpy.ndarray) -> List[int]:
            priority: List[int] = [s_index]
            for width in range(1, len(simimalities)):
                left: Optional[float] = None
                if s_index - width >= 0:
                    left = simimalities[s_index - width]
                right: Optional[float] = None
                if s_index + width < len(simimalities):
                    right = simimalities[s_index + width]
                if left is None and right is None:
                    break
                if left is None:
                    priority.append(s_index + width)
                    continue
                if right is None:
                    priority.append(s_index - width)
                    continue
                if left > right:
                    priority.append(s_index - width)
                    priority.append(s_index + width)
                    continue
                priority.append(s_index + width)
                priority.append(s_index - width)
            return priority

        def _select_valid_starts (
            tmp_starts: List[str],
            summary_priority: List[int],
            summary: SummaryResultModel,
        ) -> List[str]:
            def __select (
                tmp_starts: List[str],
                s_index: int,
                summary: SummaryResultModel,
                margin_left: int = 60,
                margin_right: int = 0,
            ) -> List[str]:
                valid_starts: List[str] = []
                time_range: Tuple[str, str] = _get_time_range(s_index, summary)
                for start in tmp_starts:
                    if _hms2s(start) < _hms2s(time_range[0]) - margin_left:
                        continue
                    if _hms2s(time_range[1]) + margin_right  < _hms2s(start):
                        continue
                    valid_starts.append(start)
                return valid_starts

            valid_starts: List[str] = []
            for s_index in summary_priority:
                valid_starts = __select(tmp_starts, s_index, summary)
                if len(valid_starts) > 0:
                    break

            return valid_starts

        def _aggregate_starts (starts: List[str]) -> List[str]:
            aggregated: List[str] = []
            idx: int = 0
            while idx < len(starts):
                if idx == len(starts) - 1:
                    aggregated.append(starts[idx])
                    break
                cur: int = _hms2s(starts[idx])
                next_idx: int = idx + 1
                while next_idx < len(starts):
                    next: int = _hms2s(starts[next_idx])
                    if next - cur > 60:
                        break
                    cur = next
                    next_idx += 1
                if next_idx - idx > 1:
                    aggregated.append(f'{starts[idx]}*')
                else:
                    aggregated.append(starts[idx])
                idx = next_idx

            return aggregated

        if vid == "":
            raise ValueError("no vid")

        if summary is None:
            summary = await YoutubeSummarize.asummary(vid)

        if summary is None:
            raise ValueError("no summary")

        if len(summary.agenda) == 0:
            return summary
        if len(summary.agenda[0].time) > 0:
            return summary

        # 各アジェンダ（タイトル＋サブタイトル）が含まれるだいたいの時間帯を取得する
        # まず各アジェンダがどの要約から生成されたのかを調べる
        # アジェンダと各要約との類似度から候補を取得して前後のつながりから調整する
        likely_summary, similarities_list = await _aget_likely_summary(summary)
        likely_summary = _fix_likely_summary(likely_summary, similarities_list)

        # 各アジェンダの開始時刻を取得する
        # 字幕原文からアジェンダで検索して候補を得る
        # 上記で調べた時間帯に含まれる候補を当該アジェンダの開始時刻として確定する
        idx = 0
        yqa = YoutubeQA(vid=vid, detail=True, ref_sources=AGENDA_TIME_TABLE_RETRIEVE_NUM)
        for agenda in summary.agenda:
            title = re.sub(r"^\d+\.?", "", agenda.title).strip()
            if len(agenda.subtitle) == 0:
                results = yqa.retrieve(title)
                tmp_starts = sorted([ result.time for result in results])
                summary_priority = _mk_summary_priority(likely_summary[idx], similarities_list[idx])
                starts = _select_valid_starts(tmp_starts, summary_priority, summary)
                starts = _aggregate_starts(starts)
                agenda.time.append(starts)
                idx += 1
                continue
            if len(agenda.subtitle) > 0:
                agenda.time.append([])
                for subtitle in agenda.subtitle:
                    a_query =  title + " " + subtitle.strip()
                    # a_query =  abstract.strip()
                    results = yqa.retrieve(a_query)
                    tmp_starts = sorted([ result.time for result in results])
                    summary_priority = _mk_summary_priority(likely_summary[idx], similarities_list[idx])
                    starts = _select_valid_starts(tmp_starts, summary_priority, summary)
                    starts = _aggregate_starts(starts)
                    agenda.time.append(starts)
                    idx += 1

        if store:
            summary_file: str = f'{os.environ["SUMMARY_STORE_DIR"]}/{vid}'
            if not os.path.isdir(os.path.dirname(summary_file)):
                os.makedirs(os.path.dirname(summary_file))
            with open(summary_file, "w") as f:
                f.write(summary.model_dump_json())

        return summary


    @classmethod
    def make (
        cls,
        vid: str = "",
        summary: Optional[SummaryResultModel] = None,
        store: bool = False,
    ) -> SummaryResultModel:
        loop = asyncio.get_event_loop()
        tasks = [cls.amake(vid, summary, store)]
        gather = asyncio.gather(*tasks)
        result: SummaryResultModel = loop.run_until_complete(gather)[0]
        return result


    @classmethod
    def print (
        cls,
        summary: Optional[SummaryResultModel] = None,
    ) -> None:
        if summary is None:
            return
        for agenda in summary.agenda:
            print(f'{agenda.title}', end="")
            if len(agenda.time[0]) > 0:
                print("  (", " ".join(agenda.time[0]), ")", sep="", end="")
            print("")
            for idx, subtitle in enumerate(agenda.subtitle):
                print(f'  {subtitle} ({", ".join(agenda.time[idx+1])})')
            print("")


class YoutubeTopicTimeTable:
    @classmethod
    async def amake (
        cls,
        vid: str = "",
        summary: Optional[SummaryResultModel] = None,
        store: bool = False,
    ) -> SummaryResultModel:
        def _cosine_similarity(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
            return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

        def _s2hms (seconds: int) -> str:
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            return "%d:%02d:%02d" % (h, m, s)

        def _hms2s (hms: str) -> int:
            h, m, s = hms.split(':')
            return int(h) * 3600 + int(m) * 60 + int(s)

        async def _aget_likely_summary (summary: SummaryResultModel) -> Tuple[List[int], List[numpy.ndarray]]:
            def __get_topic_items (summary: SummaryResultModel) -> List[str]:
                items: List[str] = []
                for topic in summary.topic:
                    content: str = topic.topic.strip()
                    items.append(content)
                return items

            llm_embedding = setup_embedding_from_environment()
            tasks = [
                llm_embedding.aembed_documents([ d.text for d in summary.detail]),
                llm_embedding.aembed_documents(__get_topic_items(summary))
            ]
            results: List[Any] = await asyncio.gather(*tasks)
            summary_embs: numpy.ndarray = numpy.array(results[0])
            topic_embs: List[numpy.ndarray] = [ numpy.array(a) for a in results[1]]

            likely_summary: List[int] = []
            similarities_list: List[numpy.ndarray] = []
            for idx in range(0, len(summary.topic)):
                similarities = numpy.array([
                    _cosine_similarity(topic_embs[idx], summary_emb) for summary_emb in summary_embs
                ])
                index = int(numpy.argmax(similarities))
                likely_summary.append(index)
                similarities_list.append(similarities)

            return (likely_summary, similarities_list)

        def _fix_likely_summary(
            likely_summary: List[int], similarities_list: List[numpy.ndarray]
        ) -> List[int]:
            def __which_is_most_similar (
                candidates_idx: List[int], similarities: numpy.ndarray
            ) -> int:
                similar_idx: int = candidates_idx[0]
                for idx in candidates_idx:
                    if idx >= len(similarities):
                        break
                    if similarities[similar_idx] < similarities[idx]:
                        similar_idx = idx
                return similar_idx

            def __force_change (
                likely_summary: List[int], similarities_list: List[numpy.ndarray]
            ) -> List[int]:
                if len(likely_summary) <= 2:
                    return likely_summary
                if len(likely_summary) < len(similarities_list[0]) + 2:
                    return likely_summary
                likely_summary[0] = 0
                likely_summary[-1] = len(similarities_list[0]) - 1
                return likely_summary


            def __compare_with_neiborhood (
                likely_summary: List[int], similarities_list: List[numpy.ndarray],
            ) -> List[int]:
                def ___get_next_summary (base_index: int, likely_summary: List[int]) -> int:
                    for idx in range(base_index+1, len(likely_summary)):
                        if likely_summary[idx] != likely_summary[base_index]:
                            return likely_summary[idx]
                    return likely_summary[base_index]

                fixed: List[int] = []
                for idx, cur_summary in enumerate(likely_summary):
                    if idx == 0 or idx == len(likely_summary) - 1:
                        fixed.append(cur_summary)
                        continue
                    similarities: numpy.ndarray = similarities_list[idx]
                    prev_summary = fixed[idx-1] if idx > 0 else 0
                    next_summary = ___get_next_summary(idx, likely_summary)
                    if prev_summary > cur_summary:
                        if prev_summary >= next_summary:
                            fixed.append(prev_summary)
                        else:
                            candidates: List[int] = [ i for i in range(prev_summary, next_summary)]
                            fixed.append(__which_is_most_similar(candidates, similarities))
                        continue
                    if cur_summary - prev_summary > 2:
                        candidates: List[int] = [prev_summary, prev_summary + 1]
                        fixed.append(__which_is_most_similar(candidates, similarities))
                        continue
                    if cur_summary - prev_summary == 2:
                        fixed.append(prev_summary + 1)
                        continue
                    candidates: List[int] = [prev_summary, prev_summary + 1]
                    fixed.append(__which_is_most_similar(candidates, similarities))

                return fixed

            def __check_sequence (likely_summary: List[int]) -> bool:
                for idx in range(len(likely_summary)):
                    cur = likely_summary[idx]
                    prev = likely_summary[idx-1] if idx > 0 else likely_summary[0]
                    next = likely_summary[idx+1] if idx + 1 < len(likely_summary) else likely_summary[-1]
                    if cur < prev or cur > next:
                        return False
                return True

            def __fix (
                likely_summary: List[int], similarities_list: List[numpy.ndarray]
            ) -> List[int]:
                likely_summary = __force_change(likely_summary, similarities_list)
                likely_summary = __compare_with_neiborhood(likely_summary, similarities_list)
                return likely_summary

            for _ in range(5):
                likely_summary = __fix(likely_summary, similarities_list)
                if __check_sequence(likely_summary):
                    break

            return likely_summary

        def _get_time_range (index: int, summary: SummaryResultModel) -> Tuple[str, str]:
            start = _s2hms(math.floor(summary.detail[index].start))
            end = _s2hms(summary.lengthSeconds)
            if index + 1 < len(summary.detail):
                end = _s2hms(math.floor(summary.detail[index + 1].start))
            return (start, end)

        def _mk_summary_priority (s_index: int, simimalities: numpy.ndarray) -> List[int]:
            priority: List[int] = [s_index]
            for width in range(1, len(simimalities)):
                left: Optional[float] = None
                if s_index - width >= 0:
                    left = simimalities[s_index - width]
                right: Optional[float] = None
                if s_index + width < len(simimalities):
                    right = simimalities[s_index + width]
                if left is None and right is None:
                    break
                if left is None:
                    priority.append(s_index + width)
                    continue
                if right is None:
                    priority.append(s_index - width)
                    continue
                if left > right:
                    priority.append(s_index - width)
                    priority.append(s_index + width)
                    continue
                priority.append(s_index + width)
                priority.append(s_index - width)
            return priority

        def _select_valid_starts (
            tmp_starts: List[str],
            summary_priority: List[int],
            summary: SummaryResultModel,
        ) -> List[str]:
            def __select (
                tmp_starts: List[str],
                s_index: int,
                summary: SummaryResultModel,
                margin_left: int = 60,
                margin_right: int = 0,
            ) -> List[str]:
                valid_starts: List[str] = []
                time_range: Tuple[str, str] = _get_time_range(s_index, summary)
                for start in tmp_starts:
                    if _hms2s(start) < _hms2s(time_range[0]) - margin_left:
                        continue
                    if _hms2s(time_range[1]) + margin_right  < _hms2s(start):
                        continue
                    valid_starts.append(start)
                return valid_starts

            valid_starts: List[str] = []
            for s_index in summary_priority:
                valid_starts = __select(tmp_starts, s_index, summary)
                if len(valid_starts) > 0:
                    break

            return valid_starts

        def _aggregate_starts (starts: List[str]) -> List[str]:
            aggregated: List[str] = []
            idx: int = 0
            while idx < len(starts):
                if idx == len(starts) - 1:
                    aggregated.append(starts[idx])
                    break
                cur: int = _hms2s(starts[idx])
                next_idx: int = idx + 1
                while next_idx < len(starts):
                    next: int = _hms2s(starts[next_idx])
                    if next - cur > 60:
                        break
                    cur = next
                    next_idx += 1
                if next_idx - idx > 1:
                    aggregated.append(f'{starts[idx]}*')
                else:
                    aggregated.append(starts[idx])
                idx = next_idx

            return aggregated

        if vid == "":
            raise ValueError("no vid")

        if summary is None:
            summary = await YoutubeSummarize.asummary(vid)

        if summary is None:
            raise ValueError("no summary")

        if len(summary.topic) == 0:
            return summary
        if len(summary.topic[0].time) > 0:
            return summary

        # 各トピックが含まれるだいたいの時間帯を取得する
        # まず各トピックがどの要約から生成されたのかを調べる
        # トピックと各要約との類似度から候補を取得して前後のつながりから調整する
        likely_summary, similarities_list = await _aget_likely_summary(summary)
        likely_summary = _fix_likely_summary(likely_summary, similarities_list)

        # 各トピックの開始時刻を取得する
        # 字幕原文からトピックで検索して開始時刻の候補を得る
        # 上記で調べた時間帯に含まれる候補を当該トピックの開始時刻として確定する
        yqa = YoutubeQA(vid=vid, detail=True, ref_sources=TOPIC_TIME_TABLE_RETRIEVE_NUM)
        for idx, topic in enumerate(summary.topic):
            content = topic.topic.strip()
            results = yqa.retrieve(content)
            tmp_starts = sorted([ result.time for result in results])
            summary_priority = _mk_summary_priority(likely_summary[idx], similarities_list[idx])
            starts = _select_valid_starts(tmp_starts, summary_priority, summary)
            starts = _aggregate_starts(starts)
            topic.time = starts

        if store:
            summary_file: str = f'{os.environ["SUMMARY_STORE_DIR"]}/{vid}'
            if not os.path.isdir(os.path.dirname(summary_file)):
                os.makedirs(os.path.dirname(summary_file))
            with open(summary_file, "w") as f:
                f.write(summary.model_dump_json())

        return summary


    @classmethod
    def make (
        cls,
        vid: str = "",
        summary: Optional[SummaryResultModel] = None,
        store: bool = False,
    ) -> SummaryResultModel:
        loop = asyncio.get_event_loop()
        tasks = [cls.amake(vid, summary, store)]
        gather = asyncio.gather(*tasks)
        result: SummaryResultModel = loop.run_until_complete(gather)[0]
        return result


    @classmethod
    def print (
        cls,
        summary: Optional[SummaryResultModel] = None,
    ) -> None:
        if summary is None:
            return
        for topic in summary.topic:
            print(f'{topic.topic}', end="")
            if len(topic.time) > 0:
                print("  (", " ".join(topic.time), ")", sep="", end="")
            print("")
