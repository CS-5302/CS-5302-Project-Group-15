# -*- coding: utf-8 -*-

"""
This script processes an audio file using different Whisper models to generate transcripts.
It demonstrates how to run Whisper models from the command line, capture their output,
and parse the resulting text and detected language.

Dependencies:
- Faster Whisper (SYSTRAN): Install via `pip install faster-whisper`
"""

from faster_whisper import WhisperModel

def transcribe_audio(audio_file, models):
    """
    Transcribes an audio file using specified Whisper models and returns the transcripts.

    :param audio_file: Path to the audio file (.mp3, .wav, .m4a, etc.)
    :param models: A list of Whisper model names to use for transcription
    :return: A dictionary containing the transcripts and detected languages for each model
    """
    transcript = {}

    for model in models:
        # Load the Whisper model (set device to "cuda" if it is available else use "cpu" - default = 'auto')
        whisper_model = WhisperModel(model)

        # Perform trancription into English
        segments, info = whisper_model.transcribe(audio_file, task='translate')

        # Extracting the detected language and transcribed text in the result
        detected_lang = info.language
        lang_prob = info.language_probability

        for segment in segments:
            text = segment.text

        # Store the detected language, language_prob, and transcribed text in the dictionary
        transcript[model] = {'lang': detected_lang, 'prob': lang_prob, 'text': text}


    return transcript

# # Example Usage

# # Specify the audio file path
# audio_file = "/kaggle/input/voice/intro.m4a"

# # List of Whisper models to use for transcription
# available_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]

# # Specify the model to use
# specified_model = ["large-v2"]

# # Transcribe the audio file using the specified model
# transcript = transcribe_audio(audio_file, specified_model)

# # Print the transcription results
# print(transcript)