import os
import getpass
import openai
from tqdm.notebook import tqdm
from uuid import uuid4 # assigns unique ID to documents
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader # caveat. SimpleDirectoryReader prefers .txt.
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
# from llama_index.llms.huggingface import HuggingFaceLLM
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
# from gradientai import Gradient

# os.environ['REPLICATE_API_TOKEN'] = getpass.getpass("REPLICATE_API_TOKEN")
# os.environ['GRADIENT_WORKSPACE_ID'] = getpass.getpass("GRADIENT_WORKSPACE_ID")
# os.environ['GRADIENT_ACCESS_TOKEN'] = getpass.getpass("GRADIENT_ACCESS_TOKEN")

class DocumentEmbeddingPipeline:

    """
    A class to manage the process of loading documents, embedding them,
    indexing with ChromaDB, and querying.
    """

    def __init__(self, model_version = "mistralai/mixtral-8x7b-instruct-v0.1", chroma_path = None, fine_tune = False):
        """
        Initialize the pipeline with the necessary configurations.

        :param model_version: Version of the machine learning model to use for embedding documents.
        :param path: Optional path for persistent storage, used for ChromaDB.
        """
        self.model_version = model_version  # Model version for document embedding
        self.fine_tune = fine_tune  # Flag for fine-tuning the model

        self.chroma_path = chroma_path  # Path for ChromaDB storage, if persistent storage is used

    def setup_nous_hermes2(instruction):
      gradient = Gradient()
      # Load the pre-fine-tuned model adapter using the saved ID or state
      model_adapter = gradient.get_model_adapter(model_adapter_id = "bea513a0-b418-4442-8ca1-c5861f851ff6_model_adapter")

      sample_query = instruction + '\n\n### Response:'
      completion = model_adapter.complete(query = sample_query, max_generated_token_count = 100).generated_output

      return completion

    def setup_lora_model(self, model_name, instructions, input):
      max_seq_length   = 2048 # Choose any! We auto support RoPE Scaling internally!
      dtype            = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
      load_in_4bit     = True # Use 4bit quantization to reduce memory usage. Can be False.
      model, tokenizer = FastLanguageModel.from_pretrained(
          model_name     = model_name, # YOUR MODEL YOU USED FOR TRAINING
          max_seq_length = max_seq_length,
          dtype          = dtype,
          load_in_4bit   = load_in_4bit,
      )

      FastLanguageModel.for_inference(model) # Enable native 2x faster inference

      alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

      ### Instruction:
      {}

      ### Input:
      {}

      ### Response:
      {}"""

      inputs = tokenizer(
      [
          alpaca_prompt.format(
              instructions, # instruction
              input, # input
              "", # output - leave this blank for generation!
          ),
      ], return_tensors = "pt").to("cuda")

      outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
      # Extract the response part
      start_index = tokenizer.batch_decode(outputs)[0].find("Response:") + len("Response:")
      end_index = tokenizer.batch_decode(outputs)[0].find("</s>", start_index)

      response_part = tokenizer.batch_decode(outputs)[0][start_index:end_index].strip()
      return response_part

    def setup_environment(self):
        """
        Setup the environment by initializing the required settings for embedding and parsing.
        """
        # Service context is a hypothetical construct that manages settings and configurations
        self.service_context = Settings  # Initialize settings for the service context
        # Initialize language model with specific configurations like model version and token limits
        if not self.fine_tune:
          self.service_context.llm = Replicate(model = self.model_version, is_chat_model = True, additional_kwargs = {"max_new_tokens": 512})
          print("GOTTEN MODEL")
        # Define the embedding model to use locally
        self.service_context.embed_model = "local:BAAI/bge-small-en-v1.5"
        # Initialize the node parser for sentence splitting with specified chunk size and overlap
        self.service_context.node_parser = SentenceSplitter()

    def prepare_documents(self, collection_name,  joining = True, persistent = False):

        """
        Load documents from a specified path and initialize a collection in ChromaDB.

        :param path: Path to the directory containing the documents to be processed.
        :param collection_name: Name of the collection to be created or used in ChromaDB.
        :param joining: If true, the documents will be joined into a single string.
        :param persistent: If true, ChromaDB will use persistent storage; otherwise, it uses ephemeral storage.
        """

        self.persistent = persistent  # Set persistent storage requirement
        # Initialize ChromaDB client based on the persistence requirement
        chroma_client = chromadb.PersistentClient(path = self.chroma_path) if persistent else chromadb.Client()
        cl = chroma_client.list_collections()
        # Check if the specified collection already exists in ChromaDB
        if collection_name in cl:
            # If the collection exists, retrieve it
            self.chroma_collection = chroma_client.get_collection(name = collection_name)
        else:
            # If the collection does not exist, create a new one with the specified name and metadata
            self.chroma_collection = chroma_client.create_collection(get_or_create = True, name = collection_name, metadata = {"hnsw:space": 'cosine'})

        idx = 0
        print("HI")
        for file in os.listdir(self.chroma_path):
            file_path = os.path.join(self.chroma_path, file)
            if file_path.endswith(('.jsonl', '.ndjson')):
                idx = idx + 1
                destination_path = os.path.join(self.chroma_path, f'output{idx}.txt')
                utils.jsonl_to_text(file_path, destination_path, 'text')

        required_exts = ['.txt']
        print(f'chroma_path = {self.chroma_path}')
        print("HELLO")
        #maryam[0], nehals[1]
        reader = SimpleDirectoryReader(
            input_dir = os.path.dirname(self.chroma_path),
            required_exts = required_exts,
            recursive = True
        )
        print("HI")
        self.documents = (reader.load_data(True))[:2]
        print(len(self.documents))


    def embed_and_index(self, model_name = "BAAI/bge-small-en-v1.5"):
        """
        Embed the documents using a specified model and index them in ChromaDB.

        :param model_name: Name of the embedding model to use for generating document embeddings.
        """
        # Initialize the embedding model
        if not self.persistent:
            Settings.embed_model = HuggingFaceEmbedding(model_name = model_name)
            # Parse and chunk documents for embedding
            chunks = self.service_context.node_parser.get_nodes_from_documents(self.documents, True)
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

        print("EMBEDDINGS DONE!!")
        # Prepare a vector store for indexing the documents in ChromaDB
        vector_store = ChromaVectorStore(chroma_collection = self.chroma_collection, add_sparse_vector = True, include_metadata = True)
        storage_context = StorageContext.from_defaults(vector_store = vector_store)

        # Create an index from the documents using the vector store and embedding model
        self.index = VectorStoreIndex.from_documents(self.documents, storage_context, True, service_context = self.service_context)
    def query_data(self, query):
        """
        Query data from the index and display the result.

        :param index: The index object to query from.
        :param query: The query string.
        """
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return response