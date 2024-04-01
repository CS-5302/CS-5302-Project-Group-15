from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import SimpleNodeParser

import pandas as pd
import os

def get_reranker(reranker_top_n, service_context):
    return LLMRerank(
            choice_batch_size = 5,
            top_n = reranker_top_n,
            service_context = service_context)

def get_retriever(documents, storage_context, service_context, K, parent_doc):

    index = VectorStoreIndex.from_documents(documents, storage_context = storage_context, service_context = service_context)
    retriever = index.as_retriever(similarity_top_k = K)

    return retriever

def join_text(directory_path):
    
    """
    Reads all CSV files in the specified directory, combines all strings in the 'NOTES' (see data_preprocessing.py) column of each CSV,
    and writes the combined string to a new text file with the same base name as the CSV file.

    :param directory_path: The directory path where the CSV files are located.
    """

    # List all files in the directory
    files_in_directory = os.listdir(directory_path)

    # Find CSV files and get the full path
    csv_files = [os.path.join(directory_path, file) for file in files_in_directory if file.endswith('.csv')]

    # Process each CSV file
    for csv_file_path in csv_files:
        # Read the CSV file
        df_csv = pd.read_csv(csv_file_path)

        # Combine all the strings in the 'NOTES' column into one single string
        combined_string = ' '.join(df_csv['NOTES'].astype(str))

        # Prepare the name of the text file (same base name as the CSV file, but with .txt extension)
        base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        txt_file_name = base_name + '.txt'
        txt_file_path = os.path.join(directory_path, txt_file_name)

        # Write the combined string to the text file
        with open(txt_file_path, 'w', encoding='utf-8') as file:
            file.write(combined_string)