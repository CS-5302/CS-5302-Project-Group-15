# -*- coding: utf-8 -*-

"""
This script processes an audio file using different Whisper models to generate transcripts.
It demonstrates how to run Whisper models from the command line, capture their output,
and parse the resulting text and detected language.

Dependencies:
- Whisper (OpenAI): Install via `pip install git+https://github.com/openai/whisper.git`
- ffmpeg: Install via `sudo apt update && sudo apt install ffmpeg`
"""

import subprocess
import re

def transcribe_audio(audio_file, models):
    """
    Transcribes an audio file using specified Whisper models and returns the transcripts.

    :param audio_file: Path to the audio file (.mp3, .wav, .m4a, etc.)
    :param models: A list of Whisper model names to use for transcription
    :return: A dictionary containing the transcripts and detected languages for each model
    """
    transcript = {}
    
    for model in models:
        # Execute the Whisper model via subprocess and capture the output
        command = subprocess.run(["whisper", audio_file, "--model", model, "--task", "translate"],
                                 capture_output=True, text=True)
        output = command.stdout

        # Regular expressions to find language and transcribed text in the output
        lang_pattern = r'Detected language: (\w+)\n\['
        text_pattern = r'\]\s*(.*)\n'

        # Search for language and text in the output
        match_language = re.search(lang_pattern, output)
        match_text = re.search(text_pattern, output)

        # Store the detected language and transcribed text in the dictionary
        # transcript[model] = {'lang': match_language.group(1), 'text': match_text.group(1)}
        transcript[model] = {'text': match_text.group(1)}


    return transcript

# Example Usage

"""
# Specify the audio file path
audio_file = "/kaggle/input/new-data/Recording (3).m4a"

# List of Whisper models to use for transcription
models = ["tiny", "base", "small", "medium", "large"]

# Transcribe the audio file using the specified models
transcript = transcribe_audio(audio_file, models)

# Print the transcription results
print(transcript)

"""
