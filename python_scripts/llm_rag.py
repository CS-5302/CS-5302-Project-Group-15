import os
import getpass
import openai
from uuid import uuid4 # assigns unique ID to documents
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader # caveat. SimpleDirectoryReader prefers .txt.
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI # resp = OpenAI().complete("Paul Graham is ")
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings # Settings.embed_model = OpenAIEmbedding()
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import PromptTemplate
from IPython.display import Markdown, display
import chromadb

# Setup environment variables
# os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
# openai.api_key = os.environ["OPENAI_API_KEY"]
# TEMP = 0.7
# Add your personal path here and comment the rest
PATH = 'C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/CS-5302-Project-Group-15/Datasets/'

# Document preparation

# Step 1: Initialize ChromaDB and get service context
chroma_client = chromadb.Client() # online
# chroma_client = chromadb.PersistentClient(path = PATH)
collection_name = 'llm_rag_medical_shaip_data'
chroma_collection = chroma_client.create_collection(name = collection_name, metadata = {"hnsw:space" : 'cosine'})
# llm = OpenAI(temperature = TEMP, model = 'gpt-4')

service_context = Settings
service_context.llm = OpenAI(model = "gpt-3.5-turbo")
service_context.embed_model = OpenAIEmbedding(model = "text-embedding-3-small")
service_context.node_parser = SentenceSplitter(chunk_size = 512, chunk_overlap = 20)
service_context.num_output = 512

service_context.context_window = 3900

# Step 2: Define Embeddings, Chunks, IDs and Metadatas

# Define Embedding Function
embed_model = HuggingFaceEmbedding(model_name = "BAAI/bge-base-en-v1.5")

# Load documents
documents = SimpleDirectoryReader(PATH).load_data()

# Chunks + embeddings + ids + metadeta creation
texts = []
chunks = service_context.node_parser(documents)

for chunk in chunks:
    texts.append(chunk.text)

text_embeds = []
metadatas = []
for text in texts:
    text_embeds.append(embed_model.get_text_embedding(text))
    metadatas.append({'source':collection_name, 'text':text})

ids = [str(uuid4()) for _ in range(len(text_embeds))]    

chroma_collection.add(
    embeddings = text_embeds,
    documents = texts,
    metadatas = metadatas,
    ids = ids
)

# Step 3: Make index in vector database and storage context
vector_store = ChromaVectorStore(chroma_collection = chroma_collection, 
                                 add_sparse_vector = True, 
                                 include_metadata = True)
storage_context = StorageContext.from_defaults(vector_store = vector_store)

# Load documents and build index
index = VectorStoreIndex.from_documents(documents, 
                                        storage_context = storage_context,
                                        embed_model = embed_model)


# Query Data from the Chroma Docker index
query_engine = index.as_query_engine()
response = query_engine.query("What is the title for Week 2 Notes?")
display(Markdown(f"<b>{response}</b>"))









# Alternative Response Generation (will come later)

# # Set Up the Query Engine and Retrieval System
# query_engine = index.as_query_engine()
# K = 3  # Number of top similar results to retrieve
# retriever = index.as_retriever(similarity_top_k = K)

# # Define the Response Synthesizer with a Custom Template
# custom_instruction = "Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know."
# template = f"{custom_instruction}\n---------------------\nWe have provided context information below.\n---------------------\n{{context_str}}\n---------------------\nGiven this information, please answer the question: {{query_str}}\n"
# qa_template = PromptTemplate(template)
# synth = get_response_synthesizer(text_qa_template = qa_template)

# # Query and Response Mechanism
# user_query = "What is the main topic of the document?"  # Example user query
# response = query_engine.query(user_query)
# print(f"Response: {response}")
