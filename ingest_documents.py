import nest_asyncio
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import os
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex

nest_asyncio.apply()

_ = load_dotenv()

Settings.embed_model = OpenAIEmbedding(
    model_name="text-embedding-3-small"
)

def ingest_pdf(file_path: str, persist_dir: str):

    print("pushing the document to the vector index")

    documents = LlamaParse(result_type="markdown").load_data(file_path)

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=persist_dir)
    index = VectorStoreIndex.from_documents(documents)
    return index

# file_path = "./doc/ai_agents_google.pdf"
# vector_db_path = "./vector_db"

# index = ingest_pdf(file_path, vector_db_path)
# query_engine = index.as_query_engine(similarity_top_k=5)
# response = query_engine.query(
#     "What are agents?"
# )
# print(response)