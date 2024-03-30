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

# Step 1:
"""
Reading the document and get service context of llm
"""

# You can find detailed instructions on how to create an entire document Ingestion Pipeline here:
# https://docs.llamaindex.ai/en/stable/examples/ingestion/document_management_pipeline/

# One idea is to test the pipeline out with 3-4 documents, which we'll also use to query the LLM. Can download part of the dataset for this.

# Step 2:
"""
Make index in vector database and storage context
"""

# This is an example code from the documentation. You should read more: https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo/
# create client and a new collection
# Creates an ephemeral Chroma client that does not persist data to disk.
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# Creates a StorageContext using the default configuration with the given vector_store
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# Query Data
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))
# Step 3:
"""
Get retrieval or make it if non-existent (most time consuming imo)
"""
K = 3 
retriever = index.as_retriever(similarity_top_k=K)
custom_instruction = """Use the following pieces of context to answer the user's question. 
                    Don't use any information outside of these pieces. If you don't know the answer, just say that you don't know, don't try to make up an answer."""

template = (
    f"{custom_instruction}"
    "---------------------\n"
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)

qa_template = PromptTemplate(template)
synth = get_response_synthesizer(text_qa_template=qa_template)

# Step 4:
"""
Bot Engine Creation (involves creation of template)
"""

# Step 5:
"""
Give Query --> Get Response
"""

# All these steps are done in the dummy code shared above