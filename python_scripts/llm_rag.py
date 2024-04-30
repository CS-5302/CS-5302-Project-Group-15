import os
import getpass
import openai
from tqdm.notebook import tqdm
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
import llama_index
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.replicate import Replicate
from python_scripts import utils

os.environ['REPLICATE_API_TOKEN'] = getpass.getpass("REPLICATE_API_TOKEN")


class DocumentEmbeddingPipeline:
   
    """
    A class to manage the process of loading documents, embedding them,
    indexing with ChromaDB, and querying.
    """

    def __init__(self, model_version = "mistralai/mixtral-8x7b-instruct-v0.1", chroma_path = None):
        """
        Initialize the pipeline with the necessary configurations.

        :param model_version: Version of the machine learning model to use for embedding documents.
        :param path: Optional path for persistent storage, used for ChromaDB.
        """
        self.model_version = model_version  # Model version for document embedding
        self.chroma_path = chroma_path  # Path for ChromaDB storage, if persistent storage is used
    

    def setup_environment(self):
        """
        Setup the environment by initializing the required settings for embedding and parsing.
        """
        # Service context is a hypothetical construct that manages settings and configurations
        self.service_context = Settings  # Initialize settings for the service context
        # Initialize language model with specific configurations like model version and token limits
        self.service_context.llm = Replicate(model = self.model_version, is_chat_model = True, additional_kwargs = {"max_new_tokens": 512})
        # Define the embedding model to use locally
        self.service_context.embed_model = "local:BAAI/bge-small-en-v1.5"
        # Initialize the node parser for sentence splitting with specified chunk size and overlap
        self.service_context.node_parser = SentenceSplitter()

    def prepare_documents(self, data_path, collection_name,  joining = True, persistent = False):

        """
        Load documents from a specified path and initialize a collection in ChromaDB.

        :param path: Path to the directory containing the documents to be processed.
        :param collection_name: Name of the collection to be created or used in ChromaDB.
        :param joining: If true, the documents will be joined into a single string.
        :param persistent: If true, ChromaDB will use persistent storage; otherwise, it uses ephemeral storage.
        """
        # Initialize ChromaDB client based on the persistence requirement
        chroma_client = chromadb.PersistentClient(path = self.chroma_path) if persistent else chromadb.Client()
        print("talha muqeem")
        cl = chroma_client.list_collections()
        print(cl)
        # Check if the specified collection already exists in ChromaDB
        if collection_name in cl:
            # If the collection exists, retrieve it
            print("collection hai")
            self.chroma_collection = chroma_client.get_collection(name = collection_name)
        else:
            print("collection nahi hai")
            # If the collection does not exist, create a new one with the specified name and metadata
            self.chroma_collection = chroma_client.create_collection(get_or_create = True, name = collection_name, metadata = {"hnsw:space": 'cosine'})  

        # if joining:
        #     utils.join_text(directory_path = data_path)
                            
        # Load documents from the given path using a simple directory reader utility
        
        # Preprocess the data if of jsonl/json type
        # print(data_path)

        for file in os.listdir(self.chroma_path):
            file_path = os.path.join(self.chroma_path, file)
            print(file_path)
            if file_path.endswith(('.jsonl', '.ndjson')):
                print("hello")
                step1 = utils.read_jsonl_to_list_of_lists(file_path)
                print("talha momy")
                step2 = utils.flatten_list_of_lists(step1)
                print("ubaid king")
                # data_path = data_path + '/json_to_text.txt'
                step3 = utils.write_list_to_file(step2, self.chroma_path)

        required_exts = ['.txt']
        print("RR")
        print(self.chroma_path)
        reader = SimpleDirectoryReader(
            input_dir = self.chroma_path,
            required_exts = required_exts,
            recursive = True
        )
        self.documents = reader.load_data(True)
        print(reader.input_files)
        print("Hi")
        print(f"loaded {len(self.documents)} docs")

    def embed_and_index(self, model_name = "BAAI/bge-small-en-v1.5"):
        """
        Embed the documents using a specified model and index them in ChromaDB.

        :param model_name: Name of the embedding model to use for generating document embeddings.
        """
        # Initialize the embedding model
        Settings.embed_model = HuggingFaceEmbedding(model_name = model_name)
        print('kkjkk')
        # Parse and chunk documents for embedding
        chunks = self.service_context.node_parser.get_nodes_from_documents(self.documents, True)
        print("chnk")
        # Initialize lists to store texts, embeddings, and metadata
        texts, text_embeds, metadatas = [], [], []

        # Iterate over chunks, embed texts, and prepare metadata
        for chunk in tqdm(chunks, desc='Chunking data'):
            texts.append(chunk.text)
            text_embeds.append(Settings.embed_model.get_text_embedding(chunk.text))
            metadatas.append({'source': self.chroma_collection.name, 'text': chunk.text})

        # Generate unique identifiers for each embedded document
        ids = [str(uuid4()) for _ in range(len(text_embeds))]

        # Add the embedded texts and metadata to the ChromaDB collection
        self.chroma_collection.add(embeddings = text_embeds, documents = texts, metadatas = metadatas, ids = ids)

        # Prepare a vector store for indexing the documents in ChromaDB
        vector_store = ChromaVectorStore(chroma_collection = self.chroma_collection, add_sparse_vector = True, include_metadata = True)
        storage_context = StorageContext.from_defaults(vector_store = vector_store)

        # Create an index from the documents using the vector store and embedding model
        self.index = VectorStoreIndex.from_documents(self.documents, storage_context, True, service_context=self.service_context)
    def query_data(self, query):
        """
        Query data from the index and display the result.

        :param index: The index object to query from.
        :param query: The query string.
        """
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return response