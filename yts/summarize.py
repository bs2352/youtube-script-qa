from typing import List, Optional
import os

from langchain.llms import OpenAI, AzureOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import openai
from youtube_transcript_api import YouTubeTranscriptApi

from .types import TranscriptChunkModel, YoutubeTranscriptType
from .utils import divide_transcriptions_into_chunks


DEFAULT_VIDEO_ID = "cEynsEWpXdA" #"Tia4YJkNlQ0" # 西園寺

MAP_PROMPT_TEMPLATE = """以下の内容を簡潔にまとめてください。:


"{text}"


簡潔な要約:"""

REDUCE_PROMPT_TEMPLATE = """以下の内容を簡潔にまとめてください。:


"{text}"


簡潔な要約:"""


class YoutubeSummarize:
    def __init__(self,
                 vid: str = DEFAULT_VIDEO_ID,
                 debug: bool = False
    ) -> None:
        self.vid: str = vid
        self.debug: bool = debug

        self.chain_type: str = 'map_reduce'
        self.llm: OpenAI|ChatOpenAI|AzureOpenAI = self._setup_llm()
        self.chunks: List[TranscriptChunkModel] = []


    def _setup_llm (self) -> OpenAI|ChatOpenAI|AzureOpenAI:
        if "OPENAI_API_KEY" in os.environ.keys():
            openai.api_key = os.environ['OPENAI_API_KEY']
            if os.environ['OPENAI_LLM_MODEL_NAME'].startswith("gpt-3.5-"):
                return ChatOpenAI(client=None, model=os.environ['OPENAI_LLM_MODEL_NAME'])
            return OpenAI(client=None, model=os.environ['OPENAI_LLM_MODEL_NAME'])

        return AzureOpenAI(
                openai_api_type=os.environ['AZURE_OPENAI_API_TYPE'],
                openai_api_base=os.environ['AZURE_OPENAI_API_BASE'],
                openai_api_version=os.environ['AZURE_OPENAI_API_VERSION'],
                openai_api_key=os.environ['AZURE_OPENAI_API_KEY'],
                model=os.environ['AZURE_LLM_MODEL_NAME'],
                deployment_name=os.environ['AZURE_LLM_DEPLOYMENT_NAME'],
                client=None
            )


    def _debug (self, message: str, end: str = "\n", flush: bool = False) -> None:
        if self.debug is False:
            return
        print(message, end=end, flush=flush)
        return


    def prepare (self) -> None:
        MAXLENGTH = 1000
        OVERLAP_LENGTH = 5
        transcriptions: List[YoutubeTranscriptType] = YouTubeTranscriptApi.get_transcript(video_id=self.vid, languages=["ja", "en"])
        self.chunks = divide_transcriptions_into_chunks(
            transcriptions,
            maxlength = MAXLENGTH,
            overlap_length = OVERLAP_LENGTH,
            id_prefix = self.vid
        )


    def run (self) -> str:
        map_prompt = PromptTemplate(template=MAP_PROMPT_TEMPLATE, input_variables=["text"])
        combine_prompt = PromptTemplate(template=REDUCE_PROMPT_TEMPLATE, input_variables=["text"])
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type=self.chain_type,
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True
        )

        docs: List[Document] = [Document(page_content=chunk.text) for chunk in self.chunks]
        summary = chain.run(docs)

        return summary
