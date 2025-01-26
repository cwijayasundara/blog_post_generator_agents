from blog_agents import agent_rag_tool, DocumentResearchAgent
import streamlit as st
import asyncio
import nest_asyncio
from pathlib import Path
import os
nest_asyncio.apply()
from ingest_documents import ingest_pdf

def sanitize_filename(filename):
    """
    Sanitize the filename to prevent path traversal attacks and remove unwanted characters.
    """
    filename = os.path.basename(filename)
    filename = "".join(c for c in filename if c.isalnum() or c in (" ", ".", "_", "-")).rstrip()
    return filename

def save_uploaded_file(uploaded_file, file_path):
    """Save the uploaded file and return the save path"""
    upload_dir = Path.cwd()/"uploaded_docs" / file_path
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_filename = Path(uploaded_file.name).name
    sanitized_filename = sanitize_filename(original_filename)
    save_path = str(upload_dir / sanitized_filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

PERSIST_DIR = "./vector_db"

async def get_answer(query):
    agent = DocumentResearchAgent(timeout=600, verbose=True)
    handler = agent.run(
        query=query,
        tools=[agent_rag_tool],
    )
    final_result = await handler
    st.write("------- Final Answer ----------\n", final_result)

st.title("Blog Post Generator")
st.write("Agentic system to generate blog posts based on research papers!.")

with st.sidebar:
    st.image("images/blog_post.jpg", width=600)
    add_radio = st.radio(
        "**Select Operation**",
        options=(
            "Step 1: upload your document",
            "Step 2: generate blog post",
        )
    )

if add_radio == "Step 1: upload your document":
    st.subheader("Upload your document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    file_uploader_button = st.button("Upload")
    if uploaded_file and file_uploader_button:
        save_path = save_uploaded_file(uploaded_file, "pdf")
        st.success(f"Document uploaded successfully: {save_path}")
        ingest_pdf(save_path, PERSIST_DIR)
        st.write("your document is successfully saved!")

elif add_radio == "Step 2: generate blog post":
    query = st.text_input("Enter your instructions to generate blog post")
    if st.button("Generate Blog Post", icon="🚀"):
        if query:
            asyncio.run(get_answer(query))
        else:
            st.warning("Please enter a query first.")





