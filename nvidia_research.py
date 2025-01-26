import nest_asyncio
from dotenv import load_dotenv
import os

nest_asyncio.apply()
_ = load_dotenv()

nvapi_key = os.getenv("NVIDIA_API_KEY")

from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.llms import ChatMessage, MessageRole

llm = NVIDIA(model="meta/llama-3.3-70b-instruct", api_key=nvapi_key)

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=("You are a helpful assistant that answers in one sentence."),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=("What are the most popular house pets in North America?"),
    ),
]

response = llm.chat(messages)

print(response)