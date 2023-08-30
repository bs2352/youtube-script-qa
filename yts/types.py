from typing import TypedDict
from pydantic import BaseModel


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