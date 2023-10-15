import os
# from llama_index import download_loader, GPTVectorStoreIndex, Document
# from youtube_transcript_api import YouTubeTranscriptApi
import openai
import dotenv
import sys


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
    import json

    vid = DEFAULT_VID
    # vid = "Tia4YJkNlQ0"
    # vid = "Bd9LWW4cxEU"
    if len(sys.argv) >= 2:
        vid = sys.argv[1]
    with open(f"./summaries/{vid}", "r") as f:
        summary = json.load(f)

    prompt_template = \
"""私はYoutube動画のアジェンダを作成しています。
以下に記載する動画のタイトルと要約からアジェンダを作成してください。
アジェンダには各セクションのタイトルのみを記載するようにしてください。
またセクションはできるだけ少なくシンプルにまとめてください。

タイトル:
{title}

要約:
{summaries}

アジェンダ:
"""

    # print(prompt_template)
    prompt = PromptTemplate(template=prompt_template, input_variables=["title", "summaries"])

    llm_chain = LLMChain(
        llm = setup_llm_from_environment(),
        prompt=prompt
    )

    summaries = ""
    for detail in summary["detail"]:
        summaries = f'{summaries}\n・{detail}'

    inputs = {
        "title": summary["title"],
        "summaries": summaries,
    }
    # print(prompt.format(**inputs))
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
        print(model, ":", count, "/", count_tokens(text))


def test_function_calling ():
    import openai
    import json

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
            "description": "Answer questions about the general content of a given video",
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

    question =  "「Azure OpenAI Developers セミナー第 2 回」というタイトルの動画から質問に回答してください。\n"
    # question += "この動画ではどのようなトピックについて話されていますか？"
    # question += "ベクトル検索とは何ですか？"
    # question += "どのような話題がありますか？"
    # question += "動画の内容を簡単に教えて"
    # question += "ベクトル検索、ハイブリッド検索、セマンティック検索の違いを教えてください。"
    question += "インストール手順を教えて"

    openai.api_key = os.environ['OPENAI_API_KEY']
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0613',
        messages=[
            {'role': 'user', 'content': question}
        ],
        functions=functions,
        function_call="auto"
    )

    message = completion['choices'][0]["message"] 
    if message["function_call"]:
        print("function", message["function_call"]["name"])
        print("function", json.loads(message["function_call"]["arguments"]))
    if message["content"]:
        print("content", message["content"])


if __name__ == "__main__":
    # get_transcription()
    # divide_topic()
    # get_topic()
    # get_topic_from_summary()
    # kmeans_embedding()
    # async_run()
    # count_tokens()
    test_function_calling()