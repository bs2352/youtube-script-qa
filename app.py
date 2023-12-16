from typing import Optional
import logging

from fastapi import FastAPI
from pydantic import BaseModel

from yts.summarize import YoutubeSummarize
from yts.types import SummaryResultModel

app = FastAPI()


class SummaryRequestModel (BaseModel):
    vid: str


@app.get (
    "/",
    summary="Hello World",
    description="Sample Top Page.",
    tags=["Top Page"]
)
async def index ():
    return {"Hello": "World"}


@app.post (
    "/summary",
    summary="Summarize Youtube video content",
    description="Please specify video ID (such as cEynsEWpXdA, nYx5UaKI8mE) for vid parameter.",
    tags=["Summary"]
)
async def summary (request_body: SummaryRequestModel):
    vid: str = request_body.vid
    try:
        summary: Optional[SummaryResultModel] = await YoutubeSummarize.asummary(vid=vid)
        if summary is None:
            raise Exception("summary not found")
    except Exception as e:
        logging.error(f"[{vid}] {str(e)}", exc_info=True)
        return {"vid": request_body.vid, "error": str(e)}, 500
    return summary


