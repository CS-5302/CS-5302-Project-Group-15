from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import SimpleNodeParser

from python_scripts import get_audio

import pandas as pd
import os
import threading

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

def record_audio():
    """
    Manages the user interaction to start and stop the audio recording.
    Collects user information and records audio, saving it in a unique file based on the user's details.
    """
    # base_path = (os.getcwd() + '\\Datasets\\Audio_Files').replace('\\', '/')
    base_path = "C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/CS-5302-Project-Group-15/Datasets/Audio_Files"
    name = input("Enter your full name: ")
    age = input("Enter your age: ")
    gender = input("Enter your gender: ")

    output_filename = get_unique_filename(base_path, name, age, gender)

    print("Press Enter to start recording.")
    input()  # Wait for Enter key to start recording

    recorder = get_audio.AudioRecorder(output_filename)
    t = threading.Thread(target = recorder.start_recording)
    t.start()

    input()  # Wait for Enter key to stop recording

    recorder.stop_recording()
    t.join()

    return output_filename


def get_unique_filename(base_path, name, age, gender):
    """
    Generates a unique filename based on the given name, age, and gender.
    If a file with the same name already exists, increments a query number.
    """
    directory = os.path.join(base_path, f"{name}_{age}_{gender}")
    os.makedirs(directory, exist_ok=True)

    query_number = 1
    while True:
        filename = f"{name}_{age}_{gender}_Query_{query_number}.wav"
        file_path = os.path.join(directory, filename)
        if not os.path.exists(file_path):
            return file_path
        query_number += 1

def generate_answer_path(original_path):
    """
    Generates a new file path by modifying the filename in the original path.
    The function replaces 'Query' with 'Query_Answer' in the filename and changes
    the file extension to '.mp3'.

    :param original_path: The original file path of the audio recording.
    :return: A new file path with the modified filename and extension.
    """
    # Split the original path into directory and filename
    directory, filename = os.path.split(original_path)

    # Replace 'Query' with 'Query_Answer' and change the extension to '.mp3' in the filename
    new_filename = filename.replace("Query", "Query_Answer").replace(".wav", ".mp3")

    # Combine the directory and the new filename to create the new path
    new_path = os.path.join(directory, new_filename)

    return new_path