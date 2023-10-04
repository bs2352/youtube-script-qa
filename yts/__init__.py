import openai
import os

if "https_proxyy" in os.environ:
    openai.proxy = os.environ["https_proxy"]
