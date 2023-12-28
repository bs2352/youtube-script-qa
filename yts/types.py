from typing import TypedDict, TypeAlias, List
from pydantic import BaseModel, Field

from langchain.llms import OpenAI, AzureOpenAI
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings


LLMType: TypeAlias = OpenAI | ChatOpenAI | AzureOpenAI | AzureChatOpenAI
EmbeddingType: TypeAlias = OpenAIEmbeddings | AzureOpenAIEmbeddings


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

class TopicModel (BaseModel):
    title: str = Field("")
    abstract: List[str] = Field([])

class SummaryResultModel (BaseModel):
    title: str = Field("")
    author: str = Field("")
    lengthSeconds: int = Field(0)
    url: str = Field("")
    concise: str = Field("")
    detail: List[str] = Field([])
    topic: List[TopicModel] = Field([])
    keyword: List[str] = Field([])

class SourceModel (BaseModel):
    id: str = Field("")
    score: float = Field(0)
    time: str = Field("")
    source: str = Field("")
