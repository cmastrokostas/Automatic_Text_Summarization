import sumy
import pytextrank
import os
import json
import string
import rouge
import spacy 
import unicodedata
import re
import torch

from SumSurvey.config import multiling_path, baseline_path, summary_path, results_path, en_path, summaries_file, huggingface_metrics
from SumSurvey.config import n_sentences, sumy_summarizers, pytextrank_summarizers
from SumSurvey.config import abstractive_models
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from datasets import load_metric
from env.rouge import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

greek_nlp = spacy.load("el_core_news_sm")
english_nlp = spacy.load("en_core_web_sm")

def summarization(language, lang_path):
    path = os.path.join(multiling_path, lang_path, baseline_path)

    # Make a List of text files in the wanted file directory.
    text_files = os.listdir(path)

    # Initialize empty set of summaries for all files.
    files_set = []

    for summarizer in sumy_summarizers:

        #Iterate through files in text files list. 
        for text_file in text_files:
            # PlaintextParser converts the document in the proper form to be summarized.
            parser = PlaintextParser.from_file(os.path.join(path, text_file), Tokenizer(language))
            text = ' '.join([str(sent) for sent in sumy_summarizers[summarizer](parser.document, n_sentences)])
            save_text(text, text_file, multiling_path, lang_path, summarizer)

    for summarizer in pytextrank_summarizers:
        for text_file in text_files:
            with open(os.path.join(path, text_file), 'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
                file1 = file.read().replace("\n\n"," ").replace("\n"," ")
            text = pyTextRank(file1, pytextrank_summarizers[summarizer], language)
            save_text(text, text_file, multiling_path, lang_path, summarizer)

    if language == 'english':
        for model in abstractive_models:
            abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(abstractive_models[model])
            abstractive_tokenizer = AutoTokenizer.from_pretrained(abstractive_models[model])

            for text_file in text_files:
                with open(os.path.join(path, text_file), 'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
                    file1 = file.read().replace("\n\n"," ").replace("\n"," ")
                text = abstractive(file1, abstractive_model, abstractive_tokenizer)
                save_text(text, text_file, multiling_path, lang_path, summarizer)
    return


def evaluation(language, lang_path):
    stemmer = Stemmer("greek")
    gr_tokenizer = Tokenizer('greek')._get_word_tokenizer('greek')
    gr_rouge = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL', 'rougeLsum'], use_stemmer = False, tokenizer = gr_tokenizer )

    # Choose the proper language for the path setup.
    language = 'english' if lang_path == en_path else 'greek'

    # Hypothesis path (machine generated) & Reference path (author assigned) summaries. 
    hyp_path = os.path.join(results_path, f"{language}_{summaries_file}")
    ref_path = os.path.join(multiling_path, lang_path, summary_path)

    # Make a List of text files in the wanted file directory.
    summary_files = os.listdir(ref_path)

    with open(hyp_path, 'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
        hypotheses_list = json.load(file)
    
    files_set = []

    # Iterate through summary files.
    for summary_file, hypothesis in zip(summary_files, hypotheses_list): 
        with open(os.path.join(ref_path, summary_file), 'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
            reference = file.read()
        hyp_file = summary_file.replace("_summary", "_baseline")

        # Metrics Included
        if language == 'english' :
            file_set = {
                summarizer:{
                    "rouge": huggingface_metrics["rouge"].compute(predictions = [hypothesis[hyp_file][summarizer]], references = [reference]),
                    "bleu-1": huggingface_metrics["bleu"].compute(predictions = [(hypothesis[hyp_file][summarizer]).split()], references = [[reference.split()]], max_order = 1),
                    "bleu-2": huggingface_metrics["bleu"].compute(predictions = [(hypothesis[hyp_file][summarizer]).split()], references = [[reference.split()]], max_order = 2),
                    "sacrebleu": huggingface_metrics["sacrebleu"].compute(predictions = [hypothesis[hyp_file][summarizer]], references = [[reference]]),
                    }
                for summarizer in sumy_summarizers|pytextrank_summarizers|abstractive_models
            }
        else:
            file_set = {
                summarizer:{
                    "rouge": gr_rouge.score(prepare(hypothesis[hyp_file][summarizer]), prepare(reference)),
                    "bleu-1": huggingface_metrics["bleu"].compute(predictions = [prepare((hypothesis[hyp_file][summarizer])).split()], references = [[prepare(reference).split()]], max_order = 1),
                    "bleu-2": huggingface_metrics["bleu"].compute(predictions = [prepare((hypothesis[hyp_file][summarizer])).split()], references = [[prepare(reference).split()]], max_order = 2),
                    "sacrebleu": huggingface_metrics["sacrebleu"].compute(predictions = [prepare(hypothesis[hyp_file][summarizer])], references = [[prepare(reference)]]),
                    }
                for summarizer in sumy_summarizers|pytextrank_summarizers
            }
        # Add each file's results in the set for all the files.
        files_set.append({summary_file.replace("_summary.txt","_body.txt"): file_set})

    # Store the results in a new file.
    with open(os.path.join(results_path, f"{language}_scores.json"),'w', encoding = 'utf8') as json_file:                       
        json.dump(files_set,json_file, ensure_ascii = False, indent = 4, separators = (',', ':'))
    return


def pyTextRank(text, summarizer, language):

    # Choose proper language model
    nlp = greek_nlp if language == "greek" else english_nlp

    # Summarization pipeline
    nlp.add_pipe(factory_name = summarizer, name = summarizer, last = True)
    doc = nlp(text)
    summary = ' '.join([str(sent) for sent in doc._.textrank.summary(limit_sentences = n_sentences)]) # Join sentences
    nlp.remove_pipe(summarizer)

    return summary

# Remove Accentation
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) 
            if unicodedata.category(c) != 'Mn')

# Greek Preprocessing
def prepare(text, stemmer = Stemmer("greek")):
    return ' '.join([strip_accents(stemmer(i)) for i in text.upper().translate(str.maketrans('', '', string.punctuation)).split()])

def abstractive(text, model, tokenizer):
    device = "cuda:0" 
    model = model.to(device)

    input_ids = tokenizer(
        [text],
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=512
    )["input_ids"].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0].to(device)

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return summary

def save_text(text, text_file, dataset, lang_path, summarizer):

    # Saved summary directories organisation: ./<dataset>/<lang>/produced/<summarizer>/<file_name>
    with open(os.path.join(dataset,lang_path, "produced", summarizer, f"{text_file}"), 'w', encoding = 'utf-8-sig', errors = 'ignore') as f :
        f.write(text)
    return