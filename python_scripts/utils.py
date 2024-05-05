from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import SimpleNodeParser

from python_scripts import get_audio

import pandas as pd
import numpy as np
import os
import threading
import wave
import pyaudio
import json
from itertools import chain
import re
from pydub import AudioSegment

import soundfile
import wave
import jsonlines


def sasti_harkat(file_path):
    """
    Re-uploads wav file
    """
    data, samplerate = soundfile.read(file_path)
    soundfile.write(file_path, data, samplerate)

def preprocess_audio(audio_path):
    """
    Preprocesses audio file
    """
    audio = AudioSegment.from_wav(audio_path)

    if audio[0] != 16000: # 16 kHz
        audio = audio.set_frame_rate(16000)
    if audio.sample_width != 2:   # int16
        audio = audio.set_sample_width(2)
    if audio.channels != 1:       # mono
        audio = audio.set_channels(1)        
    arr = np.array(audio.get_array_of_samples())
    arr = arr.astype(np.float32)/32768.0

    return arr

def get_reranker(reranker_top_n, service_context):
    """
    Returns an LLMRerank object
    """
    return LLMRerank(
            choice_batch_size = 5,
            top_n = reranker_top_n,
            service_context = service_context)

def get_retriever(documents, storage_context, service_context, K, parent_doc):
    """
    Returns a retriever object
    """
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
    # base_path = "C:/Users/Admin/OneDrive/Documents/GitHub/CS-5302-Project-Group-15/Datasets/Audio_Files"
    base_path = 'C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/CS-5302-Project-Group-15/Datasets/Audio_Files'
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


def play_wav(filename):
    """
    Plays a WAV file using the PyAudio library.

    This function opens a WAV file, initializes PyAudio, and plays the file
    in a streaming fashion.

    Parameters:
    - filename (str): The path to the WAV file.

    Note:
    This function requires the 'pyaudio' library. Ensure it is installed
    using `pip install pyaudio`.
    """
    # Open the WAV file
    wf = wave.open(filename, 'rb')

    # Create a PyAudio object
    p = pyaudio.PyAudio()

    # Open a stream on the PyAudio object
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read data in chunks
    chunk_size = 1024
    data = wf.readframes(chunk_size)

    # Play the stream chunk by chunk
    while data:
        stream.write(data)
        data = wf.readframes(chunk_size)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Close PyAudio
    p.terminate()

# Example usage of playing a WAV file
# wav_file_path = 'your_file.wav'
# play_wav(wav_file_path)

def jsonl_to_text(jsonl_file, text_file, field_name):
    """
    Converts a JSONL file to a text file by extracting a specific field from each JSON object.
    """
    with open(jsonl_file, 'r', encoding='utf-8') as input_file, open(text_file, 'w', encoding='utf-8') as output_file:
        reader = jsonlines.Reader(input_file)
        for data in reader:
            text = data.get(field_name, '')  # Change 'field_name' to the actual field name containing the text
            output_file.write(text + '\n')

# Example usage
"""
jsonl_to_text('C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/CS-5302-Project-Group-15/Datasets/MeDAL/nehal_processed.jsonl', 
              'C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/CS-5302-Project-Group-15/Datasets/MeDAL/nehal_output.txt', 
              'text')
"""


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