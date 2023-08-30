from typing import List
from collections import deque
import sys

from .types import TranscriptChunkModel, YoutubeTranscriptType



def divide_transcriptions_into_chunks (
    transcriptions: List[YoutubeTranscriptType],
    maxlength: int = 300,
    overlap_length: int = 3,
    id_prefix: str = "youtube"
) -> List[TranscriptChunkModel]:

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

    chunks: List[TranscriptChunkModel] = []
    chunk: TranscriptChunkModel | None = None
    overlaps: deque[YoutubeTranscriptType] = deque([])
    for transcription in transcriptions:
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
