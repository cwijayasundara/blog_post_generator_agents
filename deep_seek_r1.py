from llama_index.llms.deepseek import DeepSeek
from dotenv import load_dotenv
import os

_ = load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

# llm = DeepSeek(model="deepseek-reasoner", api_key=api_key)
llm = DeepSeek(model="deepseek-chat", api_key=api_key)

response = llm.complete("Is 9.9 or 9.11 bigger?")
print(response)

from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(
        role="user", content="How many 'r's are in the word 'strawberry'?"
    ),
]
resp = llm.stream_chat(messages)

for r in resp:
    print(r.delta, end="")