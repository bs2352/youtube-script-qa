import openai
import os
import dotenv

dotenv.load_dotenv()

if "https_proxy" in os.environ:
    # openai.proxy = os.environ["https_proxy"]
    os.environ["OPENAI_PROXY"] = os.environ["https_proxy"]
