# Neccessary Installations
# !pip install sentencepiece

from transformers import MarianMTModel, MarianTokenizer

import warnings
warnings.filterwarnings('ignore')

def translate_text(text, src_lang, trg_lang):

    if src_lang == trg_lang: # if target and source are the same
        return text
    else:
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

# English to Arabic
sample_text_en = "One ring to rule them all, one ring to find them, one ring to bring them all and in the darkness bind them."

"""
>>> 'خاتم واحد ليحكمهم جميعاً، خاتم واحد ليجدوهم، خاتم واحد ليجلبهم جميعاً وفي الظلام يربطهم.'
"""

# French to English
sample_text_fr = 'Aujourd’hui, on est samedi, nous rendons visite à notre grand-mère. Elle a 84 ans et elle habite à Antibes. J’adore ma grand-mère, elle est très gentille. Elle fait des bons gâteaux.'

"""
>>> 'Today, it's Saturday, we visit our grandmother. She's 84 and she lives in Antibes. I love my grandmother, she's very nice. She makes good cakes.'
"""