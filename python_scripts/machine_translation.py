# Neccessary Installations
# !pip install sentencepiece

from transformers import MarianMTModel, MarianTokenizer

import warnings
warnings.filterwarnings('ignore')

def translate_text(text, src_lang, trg_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{trg_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # tokenizing the text
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors = 'pt')

    # generating the translation
    translated = model.generate(**tokenized_text)

    # decoding the translation
    translation = tokenizer.batch_decode(translated, skip_special_tokens = True)

    return translation[0]

# Example Usage

# sample_text = "I live in Pakistan"

# # English to Arabic translation
# translated_text = translate_text(sample_text, "en", "ar")
# print(translated_text)

"""
>>> أعيش في باكستان
"""

# Note will take time for a new language as it initializes the model etc. French, Deutsch and Arabic have been done
# but accuracy has to be determined.