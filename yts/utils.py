from typing import List, Dict, Any, Type
from collections import deque
import sys
import os
import asyncio
import re

from langchain.llms import OpenAI, AzureOpenAI
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
import tiktoken

from .types import LLMType, EmbeddingType, TranscriptChunkModel, YoutubeTranscriptType


RE_TRANACRIPT_FILTER = re.compile('|'.join([
    '\\[音楽\\]', '\\[拍手\\]'
]))

END_SENTENCE_TOKENS = tuple(set([
    'です', 'ます', 'ください', 'する',
    'でした', 'ました', 'した', 'よね', 'ですね', 'けどね', 'くださいね', 'ましょうね',
    'ですよ', 'ますよ', 'ましょう', 'でしたね', 'ですね'
]))


def setup_llm_from_environment () -> LLMType:
    llm_class: Type[LLMType] = OpenAI
    llm_args: Dict[str, Any] = {
        "client": None,
        "temperature": float(os.environ['LLM_TEMPERATURE']),
        "request_timeout": int(os.environ['LLM_REQUEST_TIMEOUT']),
    }
    if "OPENAI_API_KEY" in os.environ.keys():
        llm_args = {
            **llm_args,
            "openai_api_key": os.environ['OPENAI_API_KEY'],
            "model":          os.environ['OPENAI_LLM_MODEL_NAME']
        }
        if os.environ['OPENAI_LLM_MODEL_NAME'].startswith("gpt-"):
            llm_class = ChatOpenAI
    else:
        llm_args = {
            **llm_args,
            "openai_api_type":    os.environ['AZURE_OPENAI_API_TYPE'],
            "openai_api_key":     os.environ['AZURE_OPENAI_API_KEY'],
            "azure_endpoint":     os.environ['AZURE_OPENAI_API_BASE'],
            "openai_api_version": os.environ['AZURE_LLM_OPENAI_API_VERSION'],
            "azure_deployment":   os.environ['AZURE_LLM_DEPLOYMENT_NAME'],
        }
        llm_class = AzureOpenAI
        if os.environ['AZURE_LLM_DEPLOYMENT_NAME'].startswith("gpt-"):
            llm_class = AzureChatOpenAI

    return  llm_class(**llm_args)


def setup_embedding_from_environment () -> EmbeddingType:
    llm_args: Dict[str, Any] = {
        "client": None
    }
    llm_class = OpenAIEmbeddings
    if "OPENAI_API_KEY" in os.environ.keys():
        llm_args = {
            **llm_args,
            "openai_api_key": os.environ['OPENAI_API_KEY'],
        }
    else:
        llm_args = {
            **llm_args,
            "openai_api_type":    os.environ['AZURE_OPENAI_API_TYPE'],
            "openai_api_key":     os.environ['AZURE_OPENAI_API_KEY'],
            "azure_endpoint":     os.environ['AZURE_OPENAI_API_BASE'],
            "openai_api_version": os.environ['AZURE_EMBEDDING_OPENAI_API_VERSION'],
            "azure_deployment":   os.environ['AZURE_EMBEDDING_LLM_DEPLOYMENT_NAME'],
        }
        llm_class = AzureOpenAIEmbeddings
    return llm_class(**llm_args)


def count_tokens (text: str) -> int:
    # だいたいでOK. ちゃんとしたければ下記を参照.
    # https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    model = os.environ['AZURE_LLM_DEPLOYMENT_NAME']
    if "OPENAI_API_KEY" in os.environ.keys():
        model = os.environ['OPENAI_LLM_MODEL_NAME']
    try:
        encoding = tiktoken.encoding_for_model(model.replace("35", "3.5"))
    except:
        # model not found. using cl100k_base encoding
        encoding = tiktoken.get_encoding("cl100k_base")
    count = len(encoding.encode(text))

    return count


def divide_transcriptions_into_chunks (
    transcriptions: List[YoutubeTranscriptType],
    maxlength: int = 300,
    overlap_length: int = 3,
    id_prefix: str = "youtube"
) -> List[TranscriptChunkModel]:
    def _filter_transcript (transcriptions: List[YoutubeTranscriptType]) -> List[YoutubeTranscriptType]:
        def __replace (transcript: YoutubeTranscriptType) -> YoutubeTranscriptType:
            transcript['text'] = RE_TRANACRIPT_FILTER.sub('', transcript['text'])
            return transcript
        return [__replace(t)  for t in transcriptions]

    def _overlap_chunk (overlaps: deque[YoutubeTranscriptType]) -> TranscriptChunkModel|None:
        if len(overlaps) == 0:
            return None
        new_chunk: TranscriptChunkModel = TranscriptChunkModel(id="", text="", start=0.0, duration=0.0, overlap=0)
        for s in overlaps:
            new_chunk.text += s['text']
            new_chunk.duration += s['duration']
            if new_chunk.start == 0.0:
                new_chunk.start = s['start']
        return new_chunk

    transcriptions = _filter_transcript(transcriptions)
    chunks: List[TranscriptChunkModel] = []
    chunk: TranscriptChunkModel | None = None
    overlaps: deque[YoutubeTranscriptType] = deque([])
    for transcription in transcriptions:
        # 字幕の切れ目に対して簡易的だが文末を特定して句点を追加する
        if transcription["text"].endswith(END_SENTENCE_TOKENS):
            transcription["text"] = transcription["text"] + "。"
        if chunk is None:
            chunk = TranscriptChunkModel(
                id=f"{id_prefix}-{transcription['start']}",
                text=transcription['text'],
                start=transcription['start'],
                duration=transcription['duration'],
                overlap=0
            )
        elif len(chunk.text) - chunk.overlap + len(transcription["text"]) > maxlength:
            chunks.append(chunk)
            overlap_chunk: TranscriptChunkModel | None = _overlap_chunk(overlaps)
            chunk = TranscriptChunkModel(
                id=f'{id_prefix}-{overlap_chunk.start}',
                text=overlap_chunk.text + transcription["text"],
                start=overlap_chunk.start,
                duration=overlap_chunk.duration,
                overlap=len(overlap_chunk.text)
            ) if overlap_chunk is not None else TranscriptChunkModel(
                id=f'{id_prefix}-{transcription["start"]}',
                text=transcription['text'],
                start=transcription['start'],
                duration=transcription['duration'],
                overlap=0
            )
        else:
            chunk.text += transcription["text"]
            chunk.duration += transcription["duration"]

        if overlap_length > 0:
            if len(overlaps) < overlap_length:
                overlaps.append(transcription)
            else:
                overlaps.popleft()
                overlaps.append(transcription)
    if chunk is not None:
        chunks.append(chunk)

    # for chunk in chunks:
    #     print(chunk)
    # sys.exit(0)

    return chunks


def loading_for_async_func (func):

    def _wrapper (*args, **kwargs):
        async def _loading ():
            chars = [
                '/', '―', '\\', '|', '/', '―', '\\', '|', '😍',
                '/', '―', '\\', '|', '/', '―', '\\', '|', '🤪',
                '/', '―', '\\', '|', '/', '―', '\\', '|', '😎',
            ]
            i = 0
            while i >= 0:
                i %= len(chars)
                sys.stdout.write("\033[2K\033[G %s " % chars[i])
                sys.stdout.flush()
                await asyncio.sleep(1.5)
                i += 1

        t = asyncio.ensure_future(_loading())
        res = func(*args, **kwargs)
        t.cancel()
        sys.stdout.write("\033[2K\033[G")
        sys.stdout.flush()
        return res

    return _wrapper
