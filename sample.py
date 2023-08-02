import os
from llama_index import download_loader, GPTVectorStoreIndex, Document
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import dotenv


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

MAXLENGTH = 500
scripts = YouTubeTranscriptApi.get_transcript(video_id=video_id, languages=["ja"])
for script in scripts:
    print(script)

