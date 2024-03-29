# State neccessary import + installation
# Libraries:

# !pip install llama-index chromadb
# !pip install chromadb
# !pip install sentence-transformers
# !pip install pydantic==1.10.11
# !pip install -U openai
# !pip install llama-index-storage-docstore-chroma # ChromaDB
# Imports:

# imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb

# set up OpenAI
import os

os.environ["OPENAI_API_KEY"] = "OUR_API_KEY"
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

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
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
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

# Step 4:
"""
Bot Engine Creation (involves creation of template)
"""

# Step 5:
"""
Give Query --> Get Response
"""

# All these steps are done in the dummy code shared above