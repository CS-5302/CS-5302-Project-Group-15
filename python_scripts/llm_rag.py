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
import llama_index
os.environ["REPLICATE_API_TOKEN"] = getpass.getpass("REPLICATE API KEY:")
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.replicate import Replicate




class DocumentEmbeddingPipeline:
    """
    A class to manage the process of loading documents, embedding them,
    indexing with ChromaDB, and querying.
    """
    
    def __init__(self, model_version = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5", path = None):
        """
        Initialize the pipeline with the necessary configurations.

        :param api_key: OpenAI API key for authentication.
        :param model_version: Version of the OpenAI model to use.
        :param path: Optional path for persistent ChromaDB storage.
        """
        self.model_version = model_version
        self.path = path

    def setup_environment(self):
        """
        Setup the environment by initializing OpenAI and service context.

        :return: Configured service context.
        """

        # Initialize service context with OpenAI settings
        self.service_context = Settings
        self.service_context.llm = Replicate(
                                            model=self.model_version,
                                            is_chat_model=True,
                                            additional_kwargs={"max_new_tokens": 512}
                                        )
        self.service_context.embed_model = "local:BAAI/bge-small-en-v1.5"
        self.service_context.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        self.service_context.num_output = 512
        self.service_context.context_window = 3900

    def prepare_documents(self, path, collection_name, persistent = False):
        """
        Prepare the documents by loading them and initializing the ChromaDB collection.

        :param path: Path to the directory containing the documents.
        :param collection_name: Name of the collection to create in ChromaDB.
        :param persistent: Boolean flag to indicate whether to use persistent or ephemeral ChromaDB.
        :return: Tuple of loaded documents and the initialized ChromaDB collection.
        """
        if persistent:
            chroma_client = chromadb.PersistentClient(path = self.path)
        else:
            chroma_client = chromadb.Client()
        
        self.chroma_collection = chroma_client.create_collection(name=collection_name, metadata={"hnsw:space": 'cosine'})

        self.documents = SimpleDirectoryReader(path).load_data()

    def embed_and_index(self, model_name = "BAAI/bge-base-en-v1.5"):
        """
        Embed the documents and build an index in the vector database.

        :param documents: List of documents to process.
        :param chroma_collection: ChromaDB collection object.
        :param model_name: Model name for the embedding model.
        :return: Built index object.
        """
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        chunks = self.service_context.node_parser(self.documents)

        texts, text_embeds, metadatas = [], [], []
        for chunk in chunks:
            texts.append(chunk.text)
            text_embeds.append(embed_model.get_text_embedding(chunk.text))
            metadatas.append({'source': self.chroma_collection.name, 'text': chunk.text})

        ids = [str(uuid4()) for _ in range(len(text_embeds))]

        self.chroma_collection.add(embeddings=text_embeds, documents=texts, metadatas=metadatas, ids=ids)

        vector_store = ChromaVectorStore(chroma_collection = self.chroma_collection, add_sparse_vector = True, include_metadata = True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex.from_documents(self.documents, storage_context=storage_context, embed_model = embed_model)

    def query_data(self, query):
        """
        Query data from the index and display the result.

        :param index: The index object to query from.
        :param query: The query string.
        """
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return response



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
