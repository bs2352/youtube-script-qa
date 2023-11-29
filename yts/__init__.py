import os
import dotenv

dotenv.load_dotenv()

if "https_proxy" in os.environ:
    os.environ["OPENAI_PROXY"] = os.environ["https_proxy"]
