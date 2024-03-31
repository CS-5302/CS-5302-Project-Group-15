from langdetect import detect
from gtts import gTTS
import os


def multilingual_text_to_speech(text, filename = None):
    try:
        lang = detect(text)
        print(f"Detected language: {lang}")
        tts = gTTS(text = text, lang=lang, slow=False)
        if filename:
            filepath = filename
        else:
            filepath = f"{lang}_speech.mp3" 

        directory = os.path.dirname("multilingual_tts.ipynb")  
        filepath = os.path.join(directory, filepath)
        tts.save(filepath)
        print(f"Speech saved to {filepath}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage
"""
text_en = "I am Talha Ahmed and I Live in Pakistan"  # English 
text_fr = "Bonjour tout le monde !"  # French 
text_es = "¡Hola mundo!"  # Spanish 

multilingual_text_to_speech(text_en,"")
multilingual_text_to_speech(text_fr,"")
multilingual_text_to_speech(text_es,"")

"""