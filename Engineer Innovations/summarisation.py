#nothing new just quickly deploying the summarisation model (TL;DR)
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from pathlib import Path
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import gradio as gr


#Too long didnt read



SAVED_MODEL_FOLDER = '/media/delta/S/Facebook_summarisation_model'
SAVED_TOKENIZER_FOLDER = '/media/delta/S/Facebook_summarisation_tokenizer'





if Path(SAVED_MODEL_FOLDER).exists():
    model = BartForConditionalGeneration.from_pretrained(SAVED_MODEL_FOLDER)
    print('model loaded')
else:
    print('model not found locally')
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    model.save_pretrained(SAVED_MODEL_FOLDER)
    print('model saved')

if Path(SAVED_TOKENIZER_FOLDER).exists():
    tokenizer = BartTokenizer.from_pretrained(SAVED_TOKENIZER_FOLDER)
    print('tokenizer loaded')
else:
    print('tokenizer not found locally')
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokenizer.save_pretrained(SAVED_TOKENIZER_FOLDER)
    print('tokenizer saved')

model.to(device)

def quicksummary(long_text_no_more_than_1024_words):
    long_text_no_more_than_1024_words = long_text_no_more_than_1024_words.replace('\n', ' ')
    long_text_no_more_than_1024_words = long_text_no_more_than_1024_words.rstrip(' ')
    long_text_no_more_than_1024_words = long_text_no_more_than_1024_words.lstrip(' ')
    inputs = tokenizer([long_text_no_more_than_1024_words], max_length=1024, padding='longest', return_tensors="pt").to(device)
    summary_ids = model.generate(inputs["input_ids"], num_beams=3, min_length=100, max_length=1024, early_stopping=True)
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

gr.Interface(quicksummary, "text", "text").launch()