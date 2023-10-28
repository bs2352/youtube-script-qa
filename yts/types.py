from typing import TypedDict, TypeAlias, List
from pydantic import BaseModel

from langchain.llms import OpenAI, AzureOpenAI
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI


LLMType: TypeAlias = OpenAI | ChatOpenAI | AzureOpenAI | AzureChatOpenAI

class TranscriptChunkModel (BaseModel):
    id: str
    text: str
    start: float
    duration: float
    overlap: int

class YoutubeTranscriptType (TypedDict):
    text: str
    start: float
    duration: float

class SummaryResult (TypedDict):
    title: str
    author: str
    lengthSeconds: int
    url: str
    concise: str
    detail: List[str]

