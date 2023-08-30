import os
from llama_index import download_loader, GPTVectorStoreIndex, Document
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import dotenv
import sys


INDEX_STORAGE_DIR = "./indexes"
if not os.path.isdir(INDEX_STORAGE_DIR):
    os.mkdir(INDEX_STORAGE_DIR)


dotenv.load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


# video_id = "Tia4YJkNlQ0"
video_id = "cEynsEWpXdA"

# YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
# loader = YoutubeTranscriptReader()
# documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=cEynsEWpXdA"], languages=["ja"]) # MS2
# documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=tFgqdHKsOME"], languages=["ja"]) # MS
# documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=Tia4YJkNlQ0"], languages=["ja"]) # 西園寺
# documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=oc6RV5c1yd0"])
# documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=XJRoDEctAwA"])
# for doc in documents:
#     print(doc.text, '-------------------')

# MAXLENGTH = 500
scripts = YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=["ja"])
text = ""
for script in scripts:
    # print(script)
    text += script["text"].replace("\n", " ")
    # text += script["text"]
# print(text)

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangChainDocument

# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=300,
#     chunk_overlap=20
# )
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
)
texts = text_splitter.split_text(text)
# for t in texts:
#     print(t, len(t))
#     print('----')
# sys.exit()

docs = [LangChainDocument(page_content=t) for t in texts]
# docs = [LangChainDocument(page_content=t) for i, t in enumerate(texts) if i < 3]


from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

map_prompt_template = """以下の内容を簡潔にまとめてください。:


"{text}"


簡潔な要約:"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
combine_prompt = map_prompt


chain_type = "map_reduce"
llm = OpenAI(
    client=None,
    model="text-davinci-003",
    temperature=0.0,
    # verbose=True,
    max_tokens=1024,
)
# llm = ChatOpenAI(
#     client=None,
#     model="gpt-3.5-turbo-16k",
#     temperature=0.0,
#     max_tokens=1024,
#     verbose=True,
# )
chain = load_summarize_chain(
    llm=llm,
    chain_type=chain_type,
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    verbose=True
)
summarized_text = chain.run(docs)
print(summarized_text)

# from langchain.embeddings import OpenAIEmbeddings
# from openai.embeddings_utils import cosine_similarity

# embeddings = OpenAIEmbeddings()
# doc_embeddings = embeddings.embed_documents(texts)

# prev_text = ""
# prev_embedding = [0.0e-10] * len(doc_embeddings[0])
# for text, embedding in zip(texts, doc_embeddings):
#     similarity = cosine_similarity(prev_embedding, embedding)
#     # print(text[:50], similarity, "###" if similarity < 0.80 else "")
#     if similarity < 0.80:
#         print(prev_text)
#         print(f'[{similarity}]------------')
#         print(text)
#         print('========')
#     prev_text = text
#     prev_embedding = embedding