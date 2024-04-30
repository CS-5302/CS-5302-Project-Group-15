# State neccessary import + installation
# Libraries:
import numpy as np
import os
import getpass

# Might Come in Handy

"""
(1)
from llama_index.core import Settings

# tiktoken
import tiktoken

Settings.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode

# huggingface
from transformers import AutoTokenizer

Settings.tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceH4/zephyr-7b-beta"
)

"""

"""
(2)
# Setup environment variables
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
openai.api_key = os.environ["OPENAI_API_KEY"]
TEMP = 0.7
"""

"""
(3)
For ChromaDB to easily filter out the relevant context of the user's query
collection.query(
    query_embeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":"search_string"}
)
"""
"""
(4)
# For modifying the documents in the collection, and adds them if they dont exist
collection.upsert(
    ids=["id1", "id2", "id3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    documents=["doc1", "doc2", "doc3", ...],
)
etc. (2), (3), (4) and others like it can be found here in chromaDB documentation: https://docs.trychroma.com/usage-guide
"""

"""
(5)
For getitng embeddings do model.chroma_collection.get(include = ['documents', 'embeddings', 'metadatas'])

"""

# Installations:
# !pip install llama-index chromadb
# !pip install chromadb
# !pip install sentence-transformers
# !pip install pydantic==1.10.11
# !pip install -U openai
# !pip install llama-index-storage-store-chroma 
# !pip install llama-index-llms-huggingface 
# !pip install llama-index-embeddings-huggingface
# !pip install llama_index-response-synthesizers
# !pip install llama-index-llms
# !pip install llama-index-embeddings
# pip install llama-index-llms-openai
# !pip install -U llama-index-core llama-index-llms-openai llama-index-embeddings-openai
# !pip install llama-index-llms-replicate
# pip install sounddevice numpy scipy
# pip install keyboard
# !pip install pyaudio
# !pip install audiorecorder
# !pip install streamlit-audiorecorder
# !pip install audio-recorder-streamlit
# !pip install faster-whisper
# !pip install gradio
# !pip install mistral-lang
# Imports:

from uuid import uuid4 # assigns unique ID to documents
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings # Settings.embed_model = OpenAIEmbedding()
from llama_index.core import get_response_synthesizer
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from IPython.display import Markdown, display
import chromadb

import openai