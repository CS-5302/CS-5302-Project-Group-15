# -*- coding: utf-8 -*-

# Install the necessary packages to run this code
# !pip install git+https://github.com/openai/whisper.git
# !sudo apt update && sudo apt install ffmpeg

# Import necessary modules for the code below
import subprocess
import re

# Specify audio file path (.mp3, .wav, .m4a)
audio_file = "/kaggle/input/new-data/Recording (3).m4a"

# List of available whisper models to choose from
models = ["tiny", "base", "small", "medium", "large"]

"""
A dictionary object for storing transcripts generated from each model.
User can choose to use only one model (the best one)
"""
transcript = {}

for model in models:

#     Outputs stdout from command-line
    command = subprocess.run(["whisper", audio_file, "--model", model, "--task", "translate"], capture_output=True, text=True)
    output = command.stdout

#     Patterns of interest in stdout
    lang_pattern = r'Detected language: (\w+)\n\['
    text_pattern = r'\]\s*(.*)\n'

    match_language = re.search(lang_pattern, output)
    match_text = re.search(text_pattern, output)

#     Saving the generated text as a new entry in the dictionary object
    transcript[model] = {'lang': match_language.group(1), 'text': match_text.group(1)}

print(transcript)

