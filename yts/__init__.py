import openai
import os

if "https_proxy" in os.environ:
    openai.proxy = os.environ["https_proxy"]
