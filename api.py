from typing import Optional
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from yts.summarize import YoutubeSummarize
from yts.types import SummaryResultModel
import os


STATIC_FILES_DIR = "frontend/dist"


# 起動時と停止時の処理はここに書く
# ref. https://fastapi.tiangolo.com/advanced/events/#alternative-events-deprecated
@asynccontextmanager
async def lifespan (app: FastAPI):
    # 起動時の処理
    if os.path.exists(STATIC_FILES_DIR):
        app.mount("/", StaticFiles(directory=STATIC_FILES_DIR, html=True), name="static")
    yield
    # 停止時の処理
    # ...


app = FastAPI(lifespan=lifespan)


class SummaryRequestModel (BaseModel):
    vid: str


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

