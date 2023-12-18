from typing import Optional
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
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
    # ...
    yield
    # 停止時の処理
    # ...


app = FastAPI(lifespan=lifespan)


class SummaryRequestModel (BaseModel):
    vid: str


# if os.path.exists(STATIC_FILES_DIR):
#     app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR, html=True), name="static")
@app.get (
    "/",
    summary="Top Page",
    description="Top Page.",
    tags=["Top"]
)
async def index ():
    if not os.path.exists(STATIC_FILES_DIR):
        return HTMLResponse(content="<h2>Not Found</h2>", status_code=404)
    return FileResponse(path=f"{STATIC_FILES_DIR}/index.html")


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

