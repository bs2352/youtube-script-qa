from typing import TypedDict, TypeAlias, List
from pydantic import BaseModel, Field

from langchain_openai import (
    OpenAI, ChatOpenAI, OpenAIEmbeddings, AzureOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
)

from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaIndexOpenAIEmbeddings
from llama_index.llms.azure_openai import AzureOpenAI as LlamaIndexAzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding as LlamaIndexAzureOpenAIEmbeddings


LLMType: TypeAlias = OpenAI | ChatOpenAI | AzureOpenAI | AzureChatOpenAI
EmbeddingType: TypeAlias = OpenAIEmbeddings | AzureOpenAIEmbeddings
LlamaIndexLLMType: TypeAlias = LlamaIndexOpenAI | LlamaIndexAzureOpenAI
LlamaIndexEmbeddingType: TypeAlias = LlamaIndexOpenAIEmbeddings | LlamaIndexAzureOpenAIEmbeddings


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

class DetailSummary (BaseModel):
    text: str = Field("")
    start: float = Field(0)

class AgendaModel (BaseModel):
    title: str = Field("")
    subtitle: List[str] = Field([])
    time: List[List[str]] = Field([])

class TopicModel (BaseModel):
    topic: str = Field("")
    time: str = Field("")

class SummaryResultModel (BaseModel):
    title: str = Field("")
    author: str = Field("")
    lengthSeconds: int = Field(0)
    url: str = Field("")
    concise: str = Field("")
    detail: List[DetailSummary] = Field([])
    agenda: List[AgendaModel] = Field([])
    keyword: List[str] = Field([])
    topic: List[TopicModel] = Field([])

class SourceModel (BaseModel):
    id: str = Field("")
    score: float = Field(0)
    time: str = Field("")
    source: str = Field("")
