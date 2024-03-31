from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import SimpleNodeParser


def get_reranker(reranker_top_n, service_context):
    return LLMRerank(
            choice_batch_size = 5,
            top_n = reranker_top_n,
            service_context = service_context)

def get_retriever(documents, storage_context, service_context, K, parent_doc):

    index = VectorStoreIndex.from_documents(documents, storage_context = storage_context, service_context = service_context)
    retriever = index.as_retriever(similarity_top_k = K)

    return retriever