from datasets import load_metric
import json
import string
import unicodedata
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re

from SumSurvey.config import abstractive_models


def test():
    with open(r"C:\Users\Charalampos\source\repos\Unsupervised_Text_Summarization_Survey\my_datasets\MultiLingPilot2013\en\baseline\1c124154ccb8c775fbbc4e8a7d9f5cca_baseline.txt", 'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
        file1 = file.read().replace("\n"," ")


    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

    model_name = "sshleifer/distilbart-xsum-12-6"
    for model in abstractive_models:
        print(f"{model}: ", abstractive(file1, abstractive_models[model]))
    
    print( abstractive(file1, "facebook/bart-large-xsum"))
    print("\n")


def abstractive(text,model_name):
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(
        [WHITESPACE_HANDLER(text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return summary