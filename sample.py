# pyright: reportMissingModuleSource=false

import os
# from llama_index import download_loader, GPTVectorStoreIndex, Document
# from youtube_transcript_api import YouTubeTranscriptApi
import openai
import dotenv
import sys
from typing import Optional


DEFAULT_VID = "cEynsEWpXdA"
dotenv.load_dotenv()

# INDEX_STORAGE_DIR = "./indexes"
# if not os.path.isdir(INDEX_STORAGE_DIR):
#     os.mkdir(INDEX_STORAGE_DIR)


# dotenv.load_dotenv()
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# openai.api_key = OPENAI_API_KEY


# video_id = "Tia4YJkNlQ0"
# video_id = "cEynsEWpXdA"

def get_transcription ():
    from youtube_transcript_api import YouTubeTranscriptApi

    # YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
    # loader = YoutubeTranscriptReader()
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=cEynsEWpXdA"], languages=["ja"]) # MS2
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=tFgqdHKsOME"], languages=["ja"]) # MS
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=Tia4YJkNlQ0"], languages=["ja"]) # 西園寺
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=oc6RV5c1yd0"])
    # documents = loader.load_data(ytlinks=["https://www.youtube.com/watch?v=XJRoDEctAwA"])
    # for doc in documents:
    #     print(doc.text, '-------------------')

    MAXLENGTH = 500
    vid = DEFAULT_VID
    if len(sys.argv) > 1:
        vid = sys.argv[1]
    scripts = YouTubeTranscriptApi.get_transcript(video_id=vid, languages=["ja", "en", "en-US"])
    # text = ""
    for script in scripts:
        print(script)
        # text += script["text"].replace("\n", " ")
        # text += script["text"]
    # print(text)

# from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document as LangChainDocument

# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=300,
#     chunk_overlap=20
# )
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=20,
# )
# texts = text_splitter.split_text(text)
# for t in texts:
#     print(t, len(t))
#     print('----')
# sys.exit()

# docs = [LangChainDocument(page_content=t) for t in texts]
# docs = [LangChainDocument(page_content=t) for i, t in enumerate(texts) if i < 3]


# from langchain.chains.summarize import load_summarize_chain
# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate

# map_prompt_template = """以下の内容を簡潔にまとめてください。:


# "{text}"


# 簡潔な要約:"""
# map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
# combine_prompt = map_prompt


# chain_type = "map_reduce"
# llm = OpenAI(
#     client=None,
#     model="text-davinci-003",
#     temperature=0.0,
#     # verbose=True,
#     max_tokens=1024,
# )
# llm = ChatOpenAI(
#     client=None,
#     model="gpt-3.5-turbo-16k",
#     temperature=0.0,
#     max_tokens=1024,
#     verbose=True,
# )
# chain = load_summarize_chain(
#     llm=llm,
#     chain_type=chain_type,
#     map_prompt=map_prompt,
#     combine_prompt=combine_prompt,
#     verbose=True
# )
# summarized_text = chain.run(docs)
# print(summarized_text)

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

def divide_topic ():
    from youtube_transcript_api import YouTubeTranscriptApi
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from yts.utils import divide_transcriptions_into_chunks, setup_llm_from_environment

    transcriptions = YouTubeTranscriptApi.get_transcript(video_id=DEFAULT_VID, languages=["ja", "en"])
    chunks = divide_transcriptions_into_chunks(
        transcriptions=transcriptions,
        maxlength=200,
        overlap_length=0
    )
    # for chunk in chunks:
    #     print(chunk)

#     prompt_template = \
# """以下に記載する文1と文2はYoutube動画の会話の一部を並べたもので、文2は文1に続く会話です。
# 文2は文1と同じ話題について述べられていますか？
# それとも文2は文1とは異なる新たな話題について話されていますか？
# 文2が文1と同じ話題あればYes、異なる話題であればNoと回答してください。
    prompt_template = \
"""以下に記載する文1と文2は会話の一部です。文1に続く文2で話題の変化がありますか？
変化があればであればYes、なければNoと回答してください。

文1:
{text1}

文2:
{text2}

回答:
"""
    # print(prompt_template)
    prompt = PromptTemplate(template=prompt_template, input_variables=["text1", "text2"])

    llm_chain = LLMChain(
        llm = setup_llm_from_environment(),
        prompt=prompt
    )


    prev_text_1 = ""
    prev_text_2 = ""
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            prev_text_1 = prev_text_2
            prev_text_2 = chunk.text
            continue
        inputs = {
            "text1": f'{prev_text_1.strip()}\n{prev_text_2.strip()}',
            "text2": chunk.text,
        }
        # print(prompt.format(**inputs))
        result = llm_chain.predict(**inputs).strip()
        if result == "Yes":
        # if result == "No":
            # print(prompt.format(**inputs))
            # input()
            print("#", end="", flush=True)
        else:
            print(".", end="", flush=True)
        prev_text_1 = prev_text_2
        prev_text_2 = chunk.text
        # break
        # input()
        # print(f".{result}", end="", flush=True)


def get_topic ():
    from youtube_transcript_api import YouTubeTranscriptApi
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from yts.utils import divide_transcriptions_into_chunks, setup_llm_from_environment

    transcriptions = YouTubeTranscriptApi.get_transcript(video_id=DEFAULT_VID, languages=["ja", "en"])
    chunks = divide_transcriptions_into_chunks(
        transcriptions=transcriptions,
        maxlength=1000,
        overlap_length=5
    )
    # for chunk in chunks:
    #     print(chunk)

    prompt_template = \
"""私はYoutube動画のアジェンダを作成するために動画内で触れられているトピックをリストアップしています。
「文1」には会話の一部が記載されています。
「今までのとピック」には文1までに触れられていたトピックのリストが記載されています。
「文1」と「今までトピック」からこの動画で触れられているトピックを抽出してください。

今までのトピック:
{topics}

文1:
{text}

トピック:
"""
    # print(prompt_template)
    prompt = PromptTemplate(template=prompt_template, input_variables=["topics", "text"])

    llm_chain = LLMChain(
        llm = setup_llm_from_environment(),
        prompt=prompt
    )

    topics = ""
    for idx, chunk in enumerate(chunks):
        input = {
            "topics": topics,
            "text": chunk.text,
        }
        print(prompt.format(**input))
        result = llm_chain.predict(**input)
        print(result)
        topics = result
        # break
        # input()
        # if idx > 5:
        #     break


def get_topic_from_summary ():
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from yts.utils import setup_llm_from_environment
    from yts.types import SummaryResultModel
    import json
    import dotenv

    dotenv.load_dotenv()
    vid = DEFAULT_VID
    # vid = "Tia4YJkNlQ0"
    # vid = "Bd9LWW4cxEU"
    if len(sys.argv) >= 2:
        vid = sys.argv[1] if len(sys.argv[1]) > 5 else vid

    mode = "detail"
    if len(sys.argv) == 2 and len(sys.argv[1]) <= 5:
        mode = "concise"
    if len(sys.argv) >= 3:
        mode = "concise"

    with open(f"./{os.environ['SUMMARY_STORE_DIR']}/{vid}", "r") as f:
        summary = SummaryResultModel(**(json.load(f)))

    prompt_template = \
"""私はYoutube動画のアジェンダを作成しています。
以下に記載する動画のタイトルと要約からアジェンダを作成してください。
アジェンダには各セクションのタイトルのみを記載するようにしてください。
またセクションはできるだけ少なくし、それぞれのタイトルはできるだけ簡潔にまとめてください。

タイトル:
{title}

要約:
{summaries}

アジェンダ:
"""

    prompt_template = \
"""私はYoutube動画のアジェンダを作成しています。
以下にアジェンダを作成するときの注意事項と動画のタイトルと要約を記載しています。
注意事項をよく守って、タイトルと要約からアジェンダを作成してください。

注意事項:
・アジェンダには各セクションのタイトルのみを記載するようにしてください。
・セクションはできるだけ少なくしてください。
・それぞれのセクションに付与するタイトルはできるだけ簡潔にまとめてください。

タイトル:
{title}

要約:
{summaries}

アジェンダ:
"""

    prompt_template = \
"""I am creating an agenda for Youtube videos.
Below are notes on creating an agenda, as well as video titles and summaries.
Please follow the instructions carefully and create an agenda from the title and abstract.

Notes:
- Please write only the title of each section in the agenda.
- Please keep the number of sections as small as possible.
- Please keep the titles given to each section as concise as possible.
- Please create the agenda in Japanese.

title:
{title}

summary:
{summaries}

agenda:
"""

    prompt_template = \
"""I am creating an agenda for Youtube videos.
Below are notes on creating an agenda, as well as video title and abstract.
Please follow the instructions carefully and create an agenda from the title and abstract.

Notes:
- Please create an agenda that covers the entire content of the video.
- Your agenda should include headings and a summary for each heading.
- Please include important keywords in the heading and summary whenever possible.
- Please assign each heading a sequential number such as 1, 2, 3.
- Please keep each heading as concise as possible.
- Please add a "-" to the beginning of each summary and output it as bullet points.
- Please create the summary as a subtitle, not as a sentence.
- Please keep each summary as concise as possible.
- Please create the agenda in Japanese.

title:
{title}

abstract:
{abstract}

agenda:
"""

    prompt_template = \
"""I am creating an agenda for Youtube videos.
Below are notes on creating an agenda, as well as video title, abstract and content.
Please follow the instructions carefully and create an agenda from the title, abstract and content.

Notes:
- Please create an agenda that covers the entire content of the video.
- Your agenda should include headings and some subheaddings for each heading.
- Create headings and subheadings that follow the flow of the story.
- Please include important keywords in the heading and subheading.
- Headings and subheadings should refer to the words used in the abstract and content as much as possible.
- Please include only one topic per heading or subheading.
- Please assign each heading a sequential number such as 1, 2, 3.
- Please keep each heading as concise as possible.
- Please add a "-" to the beginning of each subheading and output it as bullet points.
- Please keep each subheading as concise as possible.
- Please create the agenda in Japanese.

title:
{title}

abstract:
{abstract}

content:
{content}

agenda:
"""

    prompt_template = \
"""I am creating an agenda for Youtube videos.
Below are notes on creating an agenda, as well as video title and content.
Please follow the instructions carefully and create an agenda from the title and content.

Notes:
- Please create an agenda that covers the entire content of the video.
- Your agenda should include headings and some subheaddings for each heading.
- Create headings and subheadings that follow the flow of the story.
- Please include important keywords in the heading and subheading.
- Please include only one topic per heading or subheading.
- Please assign each heading a sequential number such as 1, 2, 3.
- Please keep each heading as concise as possible.
- Please add a "-" to the beginning of each subheading and output it as bullet points.
- Please keep each subheading as concise as possible.
- Please create the agenda in Japanese.

Title:
{title}

Content:
{content}

Agenda:
"""

    prompt_template = \
"""I am creating an agenda for Youtube videos.
Below are notes on creating an agenda, as well as video title and content.
Please follow the instructions carefully and create an agenda from the title and content.

Notes:
- Please create an agenda that covers the entire content of the video.
- Your agenda should include headings and some subheaddings for each heading.
- Create headings and subheadings that follow the flow of the story.
- Headings and subheadings should refer to the words used in the content as much as possible.
- Please include important keywords in the heading and subheading.
- Please include only one topic per heading or subheading.
- Please assign each heading a sequential number such as 1, 2, 3.
- Please keep each heading as concise as possible.
- Please add a "-" to the beginning of each subheading and output it as bullet points.
- Please keep each subheading as concise as possible.
- Please create the agenda in Japanese.

Title:
{title}

Content:
{content}

Agenda:
"""

    # print(prompt_template)
    prompt = PromptTemplate(template=prompt_template, input_variables=["title", "summaries"])

    llm_chain = LLMChain(
        llm = setup_llm_from_environment(),
        prompt=prompt
    )

    summaries = ""
    for detail in summary.detail:
        summaries = f'{summaries}\n・{detail}'
    if mode == "concise":
        summaries = summary.concise

    inputs = {
        "title": summary.title,
        # "summaries": summaries,
        # "abstract": summary["concise"],
        "content": "\n".join([ d.text for d in summary.detail]),
    }
    # print(prompt.format(**inputs))
    # print(f"mode={mode}")
    print(llm_chain.predict(**inputs))


def kmeans_embedding ():
    from youtube_transcript_api import YouTubeTranscriptApi
    import numpy as np
    from sklearn.cluster import KMeans
    from yts.utils import setup_embedding_from_environment, divide_transcriptions_into_chunks

    def _split_chunks (chunks, split_num = 5):
        total_time: float = chunks[-1].start + chunks[-1].duration
        delta: float = total_time // split_num
        splited_chunks = []
        for tc in chunks:
            idx = int(tc.start // delta)
            idx = idx if idx < split_num else split_num
            if idx + 1 > len(splited_chunks):
                splited_chunks.append([])
            splited_chunks[idx].append(tc)
        return splited_chunks

    def _make_init_cluster(splited_chunks, embeddings):
        init_cluster = []
        idx = 0
        for splited_chunk in splited_chunks:
            count = len(splited_chunk)
            cluster = []
            for i in range(idx, idx + count):
                cluster.append(embeddings[i])
            x_cluster_mean = np.mean(np.array(cluster), axis=0)
            init_cluster.append(x_cluster_mean)
            idx += count
        x_init_cluster = np.array(init_cluster)
        return x_init_cluster

    vid = DEFAULT_VID
    if len(sys.argv) >= 2:
        vid = sys.argv[1]
    transcriptions = YouTubeTranscriptApi.get_transcript(video_id=vid, languages=["ja", "en"])
    chunks = divide_transcriptions_into_chunks(
        transcriptions=transcriptions,
        maxlength=300,
        overlap_length=3,
    )
    texts = []
    for chunk in chunks:
        texts.append(chunk.text)

    llm_embedding = setup_embedding_from_environment()
    embeddings = llm_embedding.embed_documents(texts)

    splited_chunks = _split_chunks(chunks)
    x_init_cluster = _make_init_cluster(splited_chunks, embeddings)
    # print(x_init_cluster, x_init_cluster.shape)

    x_train = np.array(embeddings)

    # kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans = KMeans(n_clusters=5, random_state=0, init=x_init_cluster)
    predicted = kmeans.fit_predict(x_train)
    print(predicted)


def async_run ():
    from youtube_transcript_api import YouTubeTranscriptApi
    from yts.utils import setup_llm_from_environment, divide_transcriptions_into_chunks
    from langchain.prompts import PromptTemplate
    from langchain.schema import HumanMessage
    from langchain.chains import LLMChain
    import asyncio
    import json

    PROMPT_TEMPLATE = """以下の内容を200字以内の日本語で簡潔に要約してください。:


"{text}"


簡潔な要約:"""

    PROMPT_TEMPLATE = """以下の内容を重要な情報はできるだけ残して要約してください。:


"{text}"


要約:"""

    # async def async_generate (llm, message):
    #     results = await llm.agenerate([message])
    #     return results.generations[0][0].text

    # async def generate_concurrently (chunks):
    #     llm = setup_llm_from_environment()
    #     prompt_template = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text"])
    #     tasks = []
    #     for chunk in chunks:
    #         message = [
    #             HumanMessage(content=prompt_template.format(text=chunk.text))
    #         ]
    #         tasks.append(async_generate(llm, message))
    #     results = await asyncio.gather(*tasks)

    #     for chunk, result in zip(chunks, results):
    #         print("========\n", chunk.text, "\n------\n", result)
    #     print(len(chunks), len(results))

    async def async_generate (chain, chunk):
        answer = await chain.arun(text=chunk)
        return answer

    async def generate_concurrently (chunks):
        llm = setup_llm_from_environment()
        prompt_template = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text"])
        chain = LLMChain(llm=llm, prompt=prompt_template)
        tasks = [async_generate(chain, chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        for chunk, result in zip(chunks, results):
            print("========\n", chunk.text, "\n------\n", result)
        print(len(chunks), len(results))

    def generate_concurrently_2 (chunks):
        llm = setup_llm_from_environment()
        prompt_template = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text"])
        chain = LLMChain(llm=llm, prompt=prompt_template)
        tasks = [async_generate(chain, chunk) for chunk in chunks]
        gather = asyncio.gather(*tasks)
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(gather)
        return results


    vid = DEFAULT_VID
    if len(sys.argv) >= 2:
        vid = sys.argv[1]
    transcriptions = YouTubeTranscriptApi.get_transcript(video_id=vid, languages=["ja", "en"])
    chunks = divide_transcriptions_into_chunks(
        transcriptions=transcriptions,
        maxlength=1000,
        overlap_length=5,
    )
    # asyncio.run(generate_concurrently(chunks))
    results = generate_concurrently_2(chunks)
    for chunk, result in zip(chunks, results):
        print("========\n", chunk.text, "\n------\n", result)
    print(len(chunks), len(results))


def count_tokens ():
    import tiktoken
    from yts.utils import count_tokens

    text = "今日はオリックスvs日本ハムです。最後だし行きたかったな。"
    models = [
        "text-davinci-003",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k",
        "gpt-35-turbo",
        "gpt-35-turbo-0613",
        "gpt-35-turbo-16k",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    ]
    for model in models:
        try:
            model_x = model.replace("35", "3.5")
            encoding = tiktoken.encoding_for_model(model_x)
        except:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        # とりあえずだいたいでOK
        # https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
        count = len(encoding.encode(text))
        print(model, ":", count, "/", count_tokens(text), "/", f'{len(text)}chars')


from tenacity import retry, wait_fixed, stop_after_attempt, before_log
import logging
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(3),
    # before=before_log(logger, logging.DEBUG)
)
def test_function_calling ():
    import openai
    import json
    import asyncio

    functions = [
        {
            # "name": "answer_from_local_information",
            "name": "answer_question_about_specific_things",
            # "description": "指定された動画内で質問に関連する局所的な情報のみを利用して回答を生成する",
            # "descriotion": "Generate answers using only local information relevant to the question within the specified video",
            # "description": "指定された動画で触れられている特定の事柄に関する質問に回答する",
            "description": "Answer questions about specific things mentioned in a given video",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "動画のタイトル"
                    },
                    "question": {
                        "type": "string",
                        "description": "質問",
                    }
                },
                "required": ["title", "question"]
            }
        },
        {
            # "name": "answer_from_all_information",
            "name": "answer_question_about_general_content",
            # "description": "指定された動画内の全ての情報を利用して質問に対する回答を生成する",
            # "description": "Generates an answer to a question using all the information in the specified video",
            # "description": "指定された動画の内容全般に関する質問に回答する",
            # "description": "Answer questions about the general content of a given video",
            "description": "View the entire video and Answer questions about the general content of a given video",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "動画のタイトル",
                    },
                    "question": {
                        "type": "string",
                        "description": "質問",
                    }
                },
                "required": ["title", "question"]
            }
        },
        {
            "name": "summarize",
            "description": "summarize the content of a given video",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "動画のタイトル",
                    }
                },
                "required": ["title"],
            }
        },
    ]

    functions = [
        {
            "name": "answer_question_about_specific_things",
            # "description": "Answer questions about specific things mentioned in a given video.",
            "description": "Answer questions about specific things mentioned in a given video. Effective for questions asking what, where, when, why and how.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "動画のタイトル"
                    },
                    "question": {
                        "type": "string",
                        "description": "質問",
                    }
                },
                "required": ["title", "question"]
            }
        },
        {
            "name": "answer_question_about_general_content",
            # "description": "View the entire video and Answer questions about the general content of a given video",
            "description": "View the entire video and Answer questions about the general content of a given video. Effective for summarizing and extracting topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "動画のタイトル",
                    },
                    "question": {
                        "type": "string",
                        "description": "質問",
                    }
                },
                "required": ["title", "question"]
            }
        },
    ]

    question_prefix = "「Azure OpenAI Developers セミナー第 2 回」というタイトルの動画から次の質問に回答してください。"
    # question_prefix = "「邪馬台国はどこにあった？」というタイトルの動画から質問に回答してください。"
    questions = [
        ("この動画ではどのようなトピックについて話されていますか？", 1),
        ("この動画のトピックを箇条書きで列挙してください。", 1),
        ("動画の内容を簡単に教えて", 1),
        ("どのような話題がありますか？", 1),
        ("どのようなトピックがありますか？", 1),
        ("この動画に登場する人物を全て教えてください。", 1),
        ("ターゲットを絞って簡潔に要約してください。", 1),
        ("ベクトル検索とは何ですか？", 0),
        ("ベクトル検索、ハイブリッド検索、セマンティック検索の違いを教えてください。", 0),
        ("インストール手順を教えて", 0),
        ("邪馬台国の候補地は？", 0),
        ("一般的に邪馬台国はどこにあったと言われていますか？", 0),
    ]

    openai.api_key = os.environ['OPENAI_API_KEY']
    tasks = [
        # openai.ChatCompletion.acreate(
            # model='gpt-3.5-turbo-0613',
        openai.AsyncOpenAI().chat.completions.create (
            model='gpt-3.5-turbo-1106',
            messages=[
                {'role': 'user', 'content': f'{question_prefix}\n\n質問：{question[0]}\n'}
            ],
            functions=functions, # type: ignore
            function_call="auto",
            # temperature=0.3,
            temperature=0.0,
        )
        for question in questions
    ]
    gather = asyncio.gather(*tasks)
    # print(gather)
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(gather)

    for question, result in zip(questions, results):
        # print(result)
        # continue
        message = result.choices[0].message
        answer = message.function_call.name if hasattr(message, "function_call") else "none"
        judge = "OK" if answer == functions[question[1]]["name"] else "NG"
        print(f'{judge}\t{question[0]}\t{answer}')

    # message = completion['choices'][0]["message"]
    # if message["function_call"]:
    #     print("function", message["function_call"]["name"])
    #     print("function", json.loads(message["function_call"]["arguments"]))
    # if message["content"]:
    #     print("content", message["content"])


def qa_with_function_calling ():
    from pytube import YouTube
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import LLMResult, ChatGeneration
    from yts.utils import setup_llm_from_environment

    vid = DEFAULT_VID
    if len(sys.argv) >= 2:
        vid = sys.argv[1]
    url = f'https://www.youtube.com/watch?v={vid}'
    title = YouTube(url).vid_info["videoDetails"]["title"]

    def is_question_about_local (question):
        functions = [
            {
                "name": "answer_question_about_specific_things",
                "description": "Answer questions about specific things mentioned in a given video",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "動画のタイトル"
                        },
                        "question": {
                            "type": "string",
                            "description": "質問",
                        }
                    },
                    "required": ["title", "question"]
                }
            },
            {
                "name": "answer_question_about_general_content",
                "description": "View the entire video and Answer questions about the general content of a given video",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "動画のタイトル",
                        },
                        "question": {
                            "type": "string",
                            "description": "質問",
                        }
                    },
                    "required": ["title", "question"]
                }
            },
        ]
        # question_prefix = f"「{title}」というタイトルの動画から質問に回答してください。"
        # completion = openai.ChatCompletion.create(
        #     model='gpt-3.5-turbo-0613',
        #     messages=[
        #         {'role': 'user', 'content': f'{question_prefix}\n{question}'}
        #     ],
        #     functions=functions,
        #     function_call="auto",
        #     temperature=0.3,
        # )
        # message = completion["choices"][0]["message"] # type: ignore

        # See langchain/chains/openai_functions/openai.py
        llm = setup_llm_from_environment()
        prompt = PromptTemplate(
            template= "「{title}」というタイトルの動画から次の質問に回答してください。\n\n質問：{question}\n",
            input_variables=["title", "question"]
        )
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            llm_kwargs={
                "functions": functions
            },
            # output_parser=JsonOutputFunctionsParser(args_only=False),
            output_key="function",
            verbose=True
        )
        result: LLMResult = chain.generate([{"title":title, "question":question}])
        generation: ChatGeneration = result.generations[0][0] # type: ignore
        message = generation.message.additional_kwargs
        # print(generation)
        # print(generation.message.additional_kwargs["function_call"]["name"])
        # print(message)
        # sys.exit(0)

        func_name = functions[0]["name"]
        if "function_call" in message:
            func_name = message["function_call"]["name"]
        return True if func_name == functions[0]["name"] else False


    def answer_from_search (question):
        from yts.qa import YoutubeQA
        yqa = YoutubeQA(vid)
        return yqa.run(question)


    def answer_from_summary (question):
        import json
        summary_file = f'{os.environ["SUMMARY_STORE_DIR"]}/{vid}'
        if not os.path.exists(summary_file):
            from yts.summarize import YoutubeSummarize, MODE_DETAIL
            ys = YoutubeSummarize(vid)
            summary = ys.run(mode=MODE_DETAIL)
        else:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        summary_detail = "\n".join(summary["detail"]) # type: ignore

        llm = setup_llm_from_environment()
        prompt = PromptTemplate(
            template= "'{question}\n以下の文書の内容から回答してください。\n\n{summary_detail}\n",
            input_variables=["question", "summary_detail"]
        )
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            # verbose=True
        )
        result = chain.run(question=question, summary_detail=summary_detail)

        return result


    question = input("Query: ").strip()
    try:
        is_about_local = is_question_about_local(question)
    except:
        is_about_local = True
    # is_about_local = False
    if is_about_local:
        answer = answer_from_search(question)
        print("# local\n", f'{answer}\n')
    else:
        answer = answer_from_summary(question)
        print("# global\n", answer)


def test_loading ():
    import asyncio
    import time

    async def async_sleep ():
        sec = 0
        while sec <= 10:
            await asyncio.sleep(2)
            print(f"# {sec} sleep.")
            sec += 2
        await asyncio.sleep(2)

    async def loading ():
        chars = ['/', '-', '\\', '|', '/', '-', '\\', '|', '😍', '😎']
        # chars = ['!', '#', '$', '%', '&', '/', '-', '\\', '-', '+']
        i = 0
        while i >= 0:
            i %= len(chars)
            sys.stdout.write("\033[2K\033[G %s " % chars[i])
            sys.stdout.flush()
            await asyncio.sleep(1)
            # sys.stdout.flush()
            i += 1

    x = asyncio.ensure_future(loading())
    print(x)

    tasks = [async_sleep()]
    gather = asyncio.gather(*tasks)
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(gather)
    # loading()

    x.cancel()
    sys.stdout.write("\033[2K\033[G")
    sys.stdout.flush()


import asyncio
from yts.utils import loading_for_async_func
@loading_for_async_func
def test_decorate_loading ():
    async def func ():
        await asyncio.sleep(10)
    tasks = [func()]
    gather = asyncio.gather(*tasks)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(gather)[0]
    return "fin"


def embedding_async ():
    from langchain.embeddings import OpenAIEmbeddings
    from llama_index import GPTVectorStoreIndex, Document, ServiceContext, LLMPredictor
    from llama_index.embeddings import LangchainEmbedding
    from yts.utils import setup_llm_from_environment, setup_embedding_from_environment
    import dotenv

    dotenv.load_dotenv(".env")

    texts = [
        "オリックスバファローズ優勝するぞ！" for _ in range(0, 100)
    ]
    llm_embeddings = setup_embedding_from_environment()

    # OK
    # embedding = llm_embeddings.embed_query("オリックスバファローズ優勝するぞ！")
    # embedding = asyncio.run(llm_embeddings.aembed_query("オリックスバファローズ優勝するぞ！"))

    # OK
    # async def aget_embedding ():
    #     tasks = [llm_embeddings.aembed_query("オリックスバファローズ優勝するぞ！")]
    #     return await asyncio.gather(*tasks)
    # embedding = asyncio.run(aget_embedding())

    # OK
    # embedding = llm_embeddings.embed_documents(texts)
    # async def aembed_documents ():
    #     return await llm_embeddings.aembed_documents(texts)
    # embedding = asyncio.run(aembed_documents())

    # NG
    # async def aget_embedding_2 ():
    #     tasks = [llm_embeddings.aembed_query(text) for text in texts]
    #     return await asyncio.gather(*tasks)
    # embedding = asyncio.run(aget_embedding_2())

    # print(embedding)

    texts = [
        "オリックスバファローズ優勝するぞ！" for _ in range(0, 50)
        # "ビッグボスも頑張って欲しい。"
    ]
    documents = [
        Document(text=text.replace("\n", " "), doc_id=f"id-{idx}") for idx, text in enumerate(texts)
    ]
    dotenv.load_dotenv(".env")
    llm = setup_llm_from_environment()
    embedding = setup_embedding_from_environment()
    llm_predictor: LLMPredictor = LLMPredictor(llm=llm)
    embedding_llm: LangchainEmbedding = LangchainEmbedding(embedding)
    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor = llm_predictor,
        embed_model = embedding_llm,
    )
    index: GPTVectorStoreIndex = GPTVectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        show_progress=True,
        use_async=True, # バグ？なぜかエラーになる。
    )
    print("fin")


def which_document_to_read ():
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    import dotenv
    from yts.utils import setup_llm_from_environment

    dotenv.load_dotenv(".env")

    prompt_template = """次の質問に回答するためにはどのドキュメントを参照すれば良いですか？
「ドキュメント：」に記載されたものから参照すべきドキュメントを全て選択してください。

質問：
{question}

ドキュメント：
{documents}

選択したドキュメント：
"""

    documents = [
        "管理サーバー導入の手引き.pdf",
        "検査サーバー導入の手引き.pdf",
        "管理サーバー利用の手引き.pdf",
        "検査サーバー利用の手引き.pdf",
    ]

    llm = setup_llm_from_environment()
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "documents"]
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    documents_val = ""
    for idx, document in enumerate(documents):
        documents_val += f"{idx}: {document}\n"

    question_val = "ライセンスはどのように登録すれば良いですか？"
    question_val = "キーワード検査をするためにはどのように設定すれば良いですか？"

    result = chain.run(
        question=question_val,
        documents=documents_val
    )

    print(result)


def test_check_comprehensively ():
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from yts.utils import setup_llm_from_environment
    from yts.types import SummaryResultModel
    import json
    import dotenv
    import time

    dotenv.load_dotenv()
    vid = DEFAULT_VID
    if len(sys.argv) >= 2:
        vid = sys.argv[1] if len(sys.argv[1]) > 5 else vid

    with open(f"./{os.environ['SUMMARY_STORE_DIR']}/{vid}", "r") as f:
        summary = SummaryResultModel(**(json.load(f)))

    title = summary.title
    concise = summary.concise
    detail = "\n".join([ d.text for d in summary.detail])
    agenda = ""
    for a in summary.agenda:
        agenda += f"{a.title}\n"
        agenda += "\n".join(a.subtitle) + "\n"
    agenda = agenda.strip()

    prompt_template = """次の{target}には本文の内容が網羅的に含まれていますか？
含まれている場合はYesとだけ回答してください。
含まれていない場合はNoと網羅されていない内容を回答してください。

{target}：
{summary}

本文：
{content}

回答：
"""
    prompt_variables = ["target", "summary", "content"]

    llm = setup_llm_from_environment()
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=prompt_variables,
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )

    args = [
        {
            "target": "要約",
            "summary": concise,
            "content": detail,
        },
        {
            "target": "目次",
            "summary": agenda,
            "content": detail,
        }
    ]

    for arg in args:
        result = chain.run(**arg)
        print(result)
        time.sleep(3)

    return ""


def test_extract_keyword ():
    # from yts.summarize import get_summary
    from yts.utils import setup_llm_from_environment
    from yts.types import SummaryResultModel
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    import json

    vid = DEFAULT_VID
    if len(sys.argv) >= 2:
        vid = sys.argv[1]

    with open(f"./{os.environ['SUMMARY_STORE_DIR']}/{vid}", "r") as f:
        summary = SummaryResultModel(**(json.load(f)))


    prompt_template = \
"""以下に記載する動画のタイトルと本文から、この動画の内容を説明するキーワードを抽出してください。

タイトル：
{title}

本文：
{content}

キーワード：
"""

    prompt_template = \
"""Please extract the keywords that describe the content of this video from the title, abstract and content listed below.
Keywords refer to words or phrases within the main theme or content of this video.
Please observe the following notes when extracting keywords.

Notes:
- Please extract only the keywords that will impress the content of this video.
- Do not output similar keywords.
- Do not output the same keywords.
- Do not translate keywords into English.
- Please output one keyword in one line.

Title:
{title}

Abstract:
{abstract}

Content:
{content}

Keyword:
"""
    prompt_variables = ["title", "content", "abstract"]
    args = {
        "title": summary.title,
        "content": summary.concise,
        # "content": "\n".join(summary.detail),
        # "abstract": summary.concise,
    }

    # prompt_template = \
# """Please extract impressive keywords from the video content listed below.
# Please observe the following notes when extracting keywords.

# Notes:
# - Please extract only targeted and impressive keywords.
# - Do not output similar keywords.
# - Do not output the same keywords.
# - Do not translate keywords into English.
# - Please output one keyword in one line.

# Content:
# {content}

# Keywords:
# """
    prompt_template = \
"""Please extract impressive keywords from the video content listed below.
Please observe the following notes when extracting keywords.

Notes:
- Please select only targeted keywords.
- Please assign each keyword a sequential number such as 1, 2, 3.
- Please extract no more than 20 keywords.
- Do not output same keywords.
- Do not output Similar keywords.
- Do not translate keywords into English.
- Please output one keyword in one line.

Content:
{content}

Keywords:
"""
    prompt_variables = ["content"]

    llm = setup_llm_from_environment()
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=prompt_variables,
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )

    args = {
        "content": "\n".join([ d.text for d in summary.detail]),
    }
    result = chain.run(**args)
    print(result)

    print("--------------")

    args = {
        "content": "\n".join(reversed([ d.text for d in summary.detail])),
    }
    result_r = chain.run(**args)
    print(result_r)

    print("--------------")
    print(", ".join(summary.keyword))


def get_topic_from_summary_kwd ():
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from yts.utils import setup_llm_from_environment
    from yts.types import SummaryResultModel
    import json
    import dotenv

    dotenv.load_dotenv()
    vid = DEFAULT_VID
    if len(sys.argv) >= 2:
        vid = sys.argv[1] if len(sys.argv[1]) > 5 else vid

    with open(f"./{os.environ['SUMMARY_STORE_DIR']}/{vid}", "r") as f:
        summary = SummaryResultModel(**(json.load(f)))

    # prompt_template  = \
# """I am creating an agenda for Youtube videos.
# Below are notes on creating an agenda, as well as video title and content.
# Please follow the instructions carefully and create an agenda from the title and content.

# Notes:
# - Please create an agenda that covers the entire content of the video.
# - Your agenda should include headings and some subheaddings for each heading.
# - Please create headings and subheadings that follow the flow of the story.
# - Please include important keywords in the heading and subheading.
# - Please include only one topic per heading or subheading.
# - Please assign each heading a sequential number such as 1, 2, 3.
# - Please keep each heading as concise as possible.
# - Please add a "-" to the beginning of each subheading and output it as bullet points.
# - Please keep each subheading as concise as possible.
# - Please create the agenda in Japanese.

# Title:
# {title}

# Content:
# {content}

# Agenda:
# """
    prompt_template  = \
"""I am creating an agenda for Youtube videos.
Below are notes on creating an agenda, as well as video title and content.
Please follow the instructions carefully and create an agenda from the title and content.

Notes:
- Please create a targeted and concise agenda.
- Your agenda should include headings and some subheaddings for each heading.
- Please assign each heading a sequential number such as 1, 2, 3.
- Please add a "-" to the beginning of each subheading and output it as bullet points.
- Please keep each heading and subheading short and concise.
- Please create the agenda in Japanese.

Title:
{title}

Content:
{content}

Agenda:
"""
    prompt_template_variables = ["title", "content"]


    prompt_template_kw  = \
"""I am creating an agenda for Youtube videos.
Below are notes on creating an agenda, as well as video title and content.
Please follow the instructions carefully and create an agenda from the title and content.

Notes:
- Please create an agenda that covers the entire content of the video.
- Your agenda should include headings and some subheaddings for each heading.
- Create headings and subheadings that follow the flow of the story.
- Please include important keywords in the heading and subheading.
- Please include the keywords listed below for headings and subheadings as much as possible.
- Please include only one topic per heading or subheading.
- Please assign each heading a sequential number such as 1, 2, 3.
- Please keep each heading as concise as possible.
- Please add a "-" to the beginning of each subheading and output it as bullet points.
- Please keep each subheading as concise as possible.
- Please create the agenda in Japanese.

Title:
{title}

Content:
{content}

Keywords:
{keyword}

Agenda:
"""
    prompt_template_variables_kw = ["title", "keyword", "content"]

    llm = setup_llm_from_environment()

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=prompt_template_variables
    )
    inputs = {
        "title": summary.title,
        "content": "\n".join([ d.text for d in summary.detail]),
        # "content": summary.concise,
    }
    # print(inputs)

    prompt_kw = PromptTemplate(
        template=prompt_template_kw,
        input_variables=prompt_template_variables_kw
    )
    inputs_kw = {
        "title": summary.title,
        "content": "\n".join([ d.text for d in summary.detail]),
        "keyword": ", ".join(summary.keyword),
    }

    llm_chain = LLMChain(
        llm = llm,
        prompt=prompt,
        # verbose=True,
    )
    llm_chain_kw = LLMChain(
        llm = llm,
        prompt=prompt_kw,
        # verbose=True,
    )

    tasks = [llm_chain.arun(**inputs), llm_chain_kw.arun(**inputs_kw)]
    gather = asyncio.gather(*tasks)
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(gather)
    print(results[0])
    print("--------------------")
    # print(results[1])
    print("--------------------")
    print("\n", ", ".join(summary.keyword))


def test_function ():
    from yts.summarize import YoutubeSummarize, MODE_ALL, MODE_DETAIL
    from yts.types import SummaryResultModel

    vid = DEFAULT_VID
    if len(sys.argv) >= 2:
        vid = sys.argv[1]

    summary: Optional[SummaryResultModel] = YoutubeSummarize.summary(vid)
    YoutubeSummarize.print(summary=summary, mode=MODE_ALL&~MODE_DETAIL)


async def _atest_agenda_similarity ():
    from typing import Tuple, List, Any
    import re
    import sys
    import numpy
    import math
    from typing import Tuple, List, Any
    from yts.summarize import YoutubeSummarize
    from yts.types import SummaryResultModel
    from yts.qa import YoutubeQA
    from yts.utils import setup_embedding_from_environment

    def _cosine_similarity(a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))

    # def argmax_top_k (
    #     array: np.ndarray,
    #     k: int = 2,
    #     diff: float = 0.10
    # ) -> np.ndarray:
    #     k = len(array) if k > len(array) else k
    #     top_k: np.ndarray = np.argsort(array)[::-1][:k]
    #     if k == 1:
    #         return top_k
    #     high = array[top_k[0]]
    #     low = array[top_k[1]]
    #     results = [top_k[0]]
    #     if abs(top_k[0] - top_k[1]) == 1 and high - low < diff:
    #         results.append(top_k[1])
    #     return np.sort(np.array(results))

    def _s2hms (seconds: int) -> str:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def _hms2s (hms: str) -> int:
        h, m, s = hms.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)

    async def _aget_likely_summary (summary: SummaryResultModel) -> Tuple[List[int], List[numpy.ndarray]]:

        def _get_agenda_items (summary: SummaryResultModel) -> List[str]:
            items: List[str] = []
            for agenda in summary.agenda:
                title: str = re.sub(r"^\d+\.?", "", agenda.title).strip()
                if len(agenda.subtitle) == 0:
                    items.append(title)
                    continue
                for subtitle in agenda.subtitle:
                    items.append(f"{title} {subtitle}")
            return items

        # def _get_summary_and_similarities (
        #     query_emb: np.ndarray, summary_embs: np.ndarray
        # ) -> Tuple[int, np.ndarray]:
        #     similarities = cosine_similarity(query_emb, summary_embs)
        #     likely_summary = int(np.argmax(similarities))
        #     return (likely_summary, similarities)

        llm_embedding = setup_embedding_from_environment()
        tasks = [
            llm_embedding.aembed_documents([ d.text for d in summary.detail]),
            llm_embedding.aembed_documents(_get_agenda_items(summary))
        ]
        results: List[Any] = await asyncio.gather(*tasks)
        summary_embs: numpy.ndarray = numpy.array(results[0])
        agenda_embs: List[numpy.ndarray] = [ numpy.array(a) for a in results[1]]

        idx: int = 0
        likely_summary: List[int] = []
        similarities_list: List[numpy.ndarray] = []
        for agenda in summary.agenda:
            if len(agenda.subtitle) == 0:
                similarities = numpy.array([
                    _cosine_similarity(agenda_embs[idx], summary_emb) for summary_emb in summary_embs
                ])
                index = int(numpy.argmax(similarities))
                likely_summary.append(index)
                similarities_list.append(similarities)
                idx += 1
                continue
            for _ in agenda.subtitle:
                similarities = numpy.array([
                    _cosine_similarity(agenda_embs[idx], summary_emb) for summary_emb in summary_embs
                ])
                index = int(numpy.argmax(similarities))
                likely_summary.append(index)
                similarities_list.append(similarities)
                idx += 1

        return (likely_summary, similarities_list)

    def _fix_likely_summary(
        likely_summary: List[int], similarities_list: List[numpy.ndarray]
    ) -> List[int]:

        def __which_is_most_similar (
            candidates_idx: List[int], similarities: numpy.ndarray
        ) -> int:
            similar_idx: int = candidates_idx[0]
            for idx in candidates_idx:
                if idx >= len(similarities):
                    break
                if similarities[similar_idx] < similarities[idx]:
                    similar_idx = idx
            return similar_idx

        def __force_change (
            likely_summary: List[int], similarities_list: List[numpy.ndarray]
        ) -> List[int]:
            if len(likely_summary) <= 2:
                return likely_summary
            if len(likely_summary) < len(similarities_list[0]) + 2:
                return likely_summary
            # if likely_summary[0] not in [0, 1]:
            #     likely_summary[0] = which_is_most_similar([0, 1], similarities_list[0])
            # max_index = len(similarities_list[0]) - 1
            # if likely_summary[-1] not in [max_index - 1, max_index]:
            #     likely_summary[-1] = which_is_most_similar([max_index - 1, max_index], similarities_list[-1])
            likely_summary[0] = 0
            likely_summary[-1] = len(similarities_list[0]) - 1
            return likely_summary

        # def _compare_with_previous (
        #     likely_summary: List[int], similarities_list: List[np.ndarray],
        # ) -> List[int]:
        #     reviewed_likely_summary: List[int] = []
        #     for idx, cur_s_index in enumerate(likely_summary):
        #         similarities: np.ndarray = similarities_list[idx]
        #         # prev_s_index = likely_summary[idx-1] if idx > 0 else 0
        #         prev_s_index = reviewed_likely_summary[idx-1] if idx > 0 else 0
        #         if prev_s_index > cur_s_index:
        #             cur_s_index = _which_is_most_similar([prev_s_index, prev_s_index+1], similarities)
        #         elif cur_s_index - prev_s_index > 2:
        #             cur_s_index = _which_is_most_similar(
        #                 [prev_s_index, prev_s_index+1, prev_s_index+2], similarities
        #             )
        #         reviewed_likely_summary.append(cur_s_index)
        #     return reviewed_likely_summary

        # def _compare_with_next (
        #     likely_summary: List[int], similarities_list: List[np.ndarray]
        # ) -> List[int]:
        #     reviewed_likely_summary: List[int] = []
        #     for idx, s_index in enumerate(likely_summary):
        #         similarities: np.ndarray = similarities_list[idx]
        #         prev_s_index = likely_summary[idx - 1] if idx > 0 else 0
        #         next_s_index = likely_summary[idx + 1] if idx < len(likely_summary) - 1 else len(similarities) - 1
        #         if s_index > next_s_index:
        #             if prev_s_index == next_s_index:
        #                 s_index = prev_s_index
        #             elif prev_s_index < next_s_index:
        #                 if next_s_index - prev_s_index == 0:
        #                     s_index = prev_s_index
        #                 elif next_s_index - prev_s_index == 1:
        #                     if similarities[prev_s_index] > similarities[next_s_index]:
        #                         s_index = prev_s_index
        #                     else:
        #                         s_index = next_s_index
        #             else:
        #                 s_index = prev_s_index
        #                 if prev_s_index + 1 < len(similarities) - 1 and similarities[prev_s_index] < similarities[prev_s_index+1]:
        #                     s_index = prev_s_index + 1
        #         reviewed_likely_summary.append(s_index)
        #     return reviewed_likely_summary

        def __compare_with_neiborhood (
            likely_summary: List[int], similarities_list: List[numpy.ndarray],
        ) -> List[int]:

            def __get_next_summary (base_index: int, likely_summary: List[int]) -> int:
                for idx in range(base_index+1, len(likely_summary)):
                    if likely_summary[idx] != likely_summary[base_index]:
                        return likely_summary[idx]
                return base_index

            fixed: List[int] = []
            for idx, cur_summary in enumerate(likely_summary):
                if idx == 0 or idx == len(likely_summary) - 1:
                    fixed.append(cur_summary)
                    continue
                similarities: numpy.ndarray = similarities_list[idx]
                prev_summary = fixed[idx-1] if idx > 0 else 0
                next_summary = __get_next_summary(idx, likely_summary)
                if prev_summary > cur_summary:
                    if prev_summary >= next_summary:
                        fixed.append(prev_summary)
                    else:
                        candidates: List[int] = [ i for i in range(prev_summary, next_summary)]
                        fixed.append(__which_is_most_similar(candidates, similarities))
                    continue
                if cur_summary - prev_summary > 2:
                    # candidates: List[int] = [ i for i in range(prev_summary + 1, cur_summary)]
                    candidates: List[int] = [prev_summary, prev_summary + 1]
                    fixed.append(__which_is_most_similar(candidates, similarities))
                    # reviewed.append(prev_summary + 1)
                    continue
                if cur_summary - prev_summary == 2:
                    fixed.append(prev_summary + 1)
                    continue
                candidates: List[int] = [prev_summary, prev_summary + 1]
                fixed.append(__which_is_most_similar(candidates, similarities))

            return fixed

        def __check_sequence (likely_summary: List[int]) -> bool:
            for idx in range(len(likely_summary)):
                cur = likely_summary[idx]
                prev = likely_summary[idx-1] if idx > 0 else likely_summary[0]
                next = likely_summary[idx+1] if idx + 1 < len(likely_summary) else likely_summary[-1]
                if cur < prev or cur > next:
                    return False
            return True

        def __fix (
            likely_summary: List[int], similarities_list: List[numpy.ndarray]
        ) -> List[int]:
            likely_summary = __force_change(likely_summary, similarities_list)
            # likely_summary= _compare_with_previous(likely_summary, similarities_list)
            # likely_summary= _compare_with_next(likely_summary, similarities_list)
            likely_summary = __compare_with_neiborhood(likely_summary, similarities_list)
            return likely_summary

        for _ in range(5):
            likely_summary = __fix(likely_summary, similarities_list)
            if __check_sequence(likely_summary):
                break

        return likely_summary

    def _get_time_range (index: int, summary: SummaryResultModel) -> Tuple[str, str]:
        start = _s2hms(math.floor(summary.detail[index].start))
        end = _s2hms(summary.lengthSeconds)
        if index + 1 < len(summary.detail):
            end = _s2hms(math.floor(summary.detail[index + 1].start))
        return (start, end)

    def _mk_summary_priority (s_index: int, simimalities: numpy.ndarray) -> List[int]:
        priority: List[int] = [s_index]
        for width in range(1, len(simimalities)):
            left: Optional[float] = None
            if s_index - width >= 0:
                left = simimalities[s_index - width]
            right: Optional[float] = None
            if s_index + width < len(simimalities):
                right = simimalities[s_index + width]
            if left is None and right is None:
                break
            if left is None:
                priority.append(s_index + width)
                continue
            if right is None:
                priority.append(s_index - width)
                continue
            if left > right:
                priority.append(s_index - width)
                priority.append(s_index + width)
                continue
            priority.append(s_index + width)
            priority.append(s_index - width)
        return priority

    def _select_valid_starts (
        tmp_starts: List[str],
        summary_priority: List[int],
        summary: SummaryResultModel,
    ) -> List[str]:

        def __select (
            tmp_starts: List[str],
            s_index: int,
            summary: SummaryResultModel,
            margin_left: int = 60,
            margin_right: int = 0,
        ) -> List[str]:
            valid_starts: List[str] = []
            time_range: Tuple[str, str] = _get_time_range(s_index, summary)
            for start in tmp_starts:
                if _hms2s(start) < _hms2s(time_range[0]) - margin_left:
                    continue
                if _hms2s(time_range[1]) + margin_right  < _hms2s(start):
                    continue
                valid_starts.append(start)
            return valid_starts

        valid_starts: List[str] = []
        for s_index in summary_priority:
            valid_starts = __select(tmp_starts, s_index, summary)
            if len(valid_starts) > 0:
                break

        return valid_starts

    def _aggregate_starts (starts: List[str]) -> List[str]:
        aggregated: List[str] = []
        idx: int = 0
        while idx < len(starts):
            if idx == len(starts) - 1:
                aggregated.append(starts[idx])
                break
            cur: int = _hms2s(starts[idx])
            next_idx: int = idx + 1
            while next_idx < len(starts):
                next: int = _hms2s(starts[next_idx])
                if next - cur > 60:
                    break
                cur = next
                next_idx += 1
            if next_idx - idx > 1:
                aggregated.append(f'{starts[idx]}*')
            else:
                aggregated.append(starts[idx])
            idx = next_idx

        return aggregated

    vid = DEFAULT_VID
    if len(sys.argv) >= 2:
        vid = sys.argv[1]

    summary: Optional[SummaryResultModel] = YoutubeSummarize.summary(vid)
    if summary is None:
        return
    # llm_embedding = setup_embedding_from_environment()
    # summary_embeddings = np.array(llm_embedding.embed_documents([ d.text for d in summary.detail]))

    likely_summary, similarities_list = await _aget_likely_summary(summary)
    # print(likely_summary)
    likely_summary = _fix_likely_summary(likely_summary, similarities_list)
    # print(likely_summary)
    # return

    idx = 0
    yqa = YoutubeQA(vid=vid, detail=True, ref_sources=5)
    for agenda in summary.agenda:
        title = re.sub(r"^\d+\.?", "", agenda.title).strip()
        if len(agenda.subtitle) == 0:
            results = yqa.retrieve(title)
            tmp_starts = sorted([ result.time for result in results])
            summary_priority = _mk_summary_priority(likely_summary[idx], similarities_list[idx])
            starts = _select_valid_starts(tmp_starts, summary_priority, summary)
            starts = _aggregate_starts(starts)
            agenda.time.append(starts)
            idx += 1
            # print("## ", agenda.title)
            # print(tmp_starts)
            # print(summary_priority)
            # print(starts)
            # print("")
            continue
        if len(agenda.subtitle) > 0:
            agenda.time.append([])
            for subtitle in agenda.subtitle:
                a_query =  title + " " + subtitle.strip()
                # a_query =  abstract.strip()
                results = yqa.retrieve(a_query)
                tmp_starts = sorted([ result.time for result in results])
                summary_priority = _mk_summary_priority(likely_summary[idx], similarities_list[idx])
                starts = _select_valid_starts(tmp_starts, summary_priority, summary)
                starts = _aggregate_starts(starts)
                agenda.time.append(starts)
                idx += 1
                # print("## ", agenda.title, subtitle)
                # print(tmp_starts)
                # print(summary_priority)
                # print(starts)
                # print("")

    return summary

def test_agenda_similarity ():
    import asyncio
    import json
    from yts.types import SummaryResultModel

    # asyncio.run(_atest_agenda_similarity())
    loop = asyncio.get_event_loop()
    tasks = [_atest_agenda_similarity()]
    gather = asyncio.gather(*tasks)
    result: SummaryResultModel = loop.run_until_complete(gather)[0]
    print(result.model_dump_json())
    print("")


def test_get_info ():
    from pytube import YouTube
    vid = DEFAULT_VID
    if len(sys.argv) >= 2:
        vid = sys.argv[1]
    url: str = f'https://www.youtube.com/watch?v={vid}'
    vinfo = YouTube(url).vid_info["videoDetails"]
    print(vinfo)


if __name__ == "__main__":
    get_transcription()
    # divide_topic()
    # get_topic()
    # get_topic_from_summary()
    # kmeans_embedding()
    # async_run()
    # count_tokens()
    # test_function_calling()
    # qa_with_function_calling()
    # test_loading()
    # print(test_decorate_loading())
    # embedding_async()
    # which_document_to_read()
    # test_check_comprehensively()
    # test_extract_keyword()
    # get_topic_from_summary_kwd()
    # test_function()
    # test_agenda_similarity()
    # test_get_info()