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




def setup_environment(api_key, model_version = "gpt-3.5-turbo"):
    """
    Setup the environment by initializing OpenAI and service context.
    
    :param api_key: OpenAI API Key for authentication
    :param model_version: Version of the OpenAI model to use
    :return: Configured service context
    """
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Initialize service context with OpenAI settings
    service_context = Settings
    service_context.llm = OpenAI(model=model_version)
    service_context.embed_model = OpenAIEmbedding(model = "text-embedding-3-small")
    service_context.node_parser = SentenceSplitter(chunk_size = 512, chunk_overlap = 20)
    service_context.num_output = 512
    service_context.context_window = 3900

    return service_context

def prepare_documents(path, collection_name, persistent = False, PATH = None):
    """
    Prepare the documents by loading them and initializing the ChromaDB collection.

    :param path: Path to the directory containing the documents
    :param collection_name: Name of the collection to create in ChromaDB
    :param persistent: Boolean flag to indicate whether to use persistent or ephemeral ChromaDB
    :param PATH: Path to the directory containing the ChromaDB data. None if persistent = False
    :return: Tuple of loaded documents and the initialized ChromaDB collection
    """
    
    if persistent:
        # Intialize chromaDB from memory
        chroma_client = chromadb.PersistentClient(path = PATH)
    else:
        # Initialize ChromaDB client and create a new collection
        chroma_client = chromadb.Client()
        chroma_collection = chroma_client.create_collection(name = collection_name, metadata = {"hnsw:space": 'cosine'})
    
    # Load documents from the specified directory
    documents = SimpleDirectoryReader(path).load_data()
    return documents, chroma_collection

def embed_and_index(documents, chroma_collection, model_name = "BAAI/bge-base-en-v1.5"):
    """
    Embed the documents and build an index in the vector database.

    :param documents: List of documents to process
    :param chroma_collection: ChromaDB collection object
    :param model_name: Model name for the embedding model
    :return: Built index object
    """
    embed_model = HuggingFaceEmbedding(model_name = model_name)
    chunks = service_context.node_parser(documents)

    # Process each chunk of the document
    texts, text_embeds, metadatas = [], [], []
    for chunk in chunks:
        texts.append(chunk.text)
        text_embeds.append(embed_model.get_text_embedding(chunk.text))
        metadatas.append({'source': collection_name, 'text': chunk.text})

    ids = [str(uuid4()) for _ in range(len(text_embeds))]

    # Add processed data to the ChromaDB collection
    chroma_collection.add(embeddings=text_embeds, documents=texts, metadatas=metadatas, ids=ids)

    # Create a vector store and a storage context for indexing
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection, add_sparse_vector=True, include_metadata=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build the index using the processed documents
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    return index

def query_data(index, query):
    """
    Query data from the index and display the result.

    :param index: The index object to query from
    :param query: The query string
    """
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    display(Markdown(f"<b>{response}</b>"))


























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
