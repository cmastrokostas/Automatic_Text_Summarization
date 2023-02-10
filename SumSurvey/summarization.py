import sumy
import os
import json
import string
import spacy 
import re
import pandas as pd
import pytextrank
import torch
import pathlib

from SumSurvey.config import dataset_path, results_path, en_path
from SumSurvey.config import n_sentences, sumy_summarizers, pytextrank_summarizers, huggingface_metrics
from SumSurvey.config import abstractive_models, greek_abstractive_models
from SumSurvey.utils import strip_accents, prepare, save_text, pyTextRank, score
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from env.rouge import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def summarization(language, lang_path):
    path = os.path.join( dataset_path, lang_path, "baseline")

    # Make a List of text files in the wanted file directory.
    text_files = os.listdir(path)

    # Initialize empty set of summaries for all files.
    files_set = []

    print(language + ' start')
    for summarizer in sumy_summarizers:
        print(summarizer + ' start')

        #Create destination folder if it doesn't exist.
        pathlib.Path(os.path.join(dataset_path, lang_path, 'produced', summarizer)).mkdir(parents = True, exist_ok = True)
        
        #Iterate through files in text files list. 
        for text_file in text_files:
            # PlaintextParser converts the document in the proper form to be summarized.
            parser = PlaintextParser.from_file(os.path.join(path, text_file), Tokenizer(language))
            text = ' '.join([str(sent) for sent in sumy_summarizers[summarizer](parser.document, n_sentences)])
            save_text(text, text_file, dataset_path, lang_path, summarizer)

    for summarizer in pytextrank_summarizers:

        #Create destination folder if it doesn't exist.
        pathlib.Path(os.path.join(dataset_path, lang_path, 'produced', summarizer)).mkdir(parents = True, exist_ok = True)
        print(summarizer + ' start')
        for text_file in text_files:
            print(text_file)
            with open(os.path.join(path, text_file), 'r', encoding = 'utf-8-sig') as file:
                file1 = file.read().replace("\n\n", " ").replace("\n", " ")
            text = pyTextRank(file1, pytextrank_summarizers[summarizer], language)  
            save_text(text, text_file, dataset_path, lang_path, summarizer)

    if language == 'english':
        for model_name in abstractive_models:
            print(model_name + ' start')
            pathlib.Path(os.path.join(dataset_path, lang_path, 'produced', model_name)).mkdir(parents = True, exist_ok = True)
            abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(abstractive_models[model_name])

            abstractive_tokenizer = AutoTokenizer.from_pretrained(abstractive_models[model_name])

            for text_file in text_files:
                with open(os.path.join(path, text_file), 'r', encoding = 'utf-8-sig') as file:
                    file1 = file.read().replace("\n\n", " ").replace("\n", " ")
                text = abstractive(file1, abstractive_model, model_name, abstractive_tokenizer)
                save_text(text, text_file, dataset_path, lang_path, model_name)

    else :
        for model_name in greek_abstractive_models:
            print(model_name + ' start')
            pathlib.Path(os.path.join(dataset_path, lang_path, 'produced', model_name)).mkdir(parents = True, exist_ok = True)
            abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(greek_abstractive_models[model_name])
            abstractive_tokenizer = AutoTokenizer.from_pretrained(greek_abstractive_models[model_name])

            for text_file in text_files:
                with open(os.path.join(path, text_file), 'r', encoding = 'utf-8-sig') as file:
                    file1 = file.read().replace("\n\n", " ").replace("\n", " ")
                text = abstractive(file1, abstractive_model, model_name, abstractive_tokenizer)
                save_text(text, text_file, dataset_path, lang_path, model_name)
    return


def abstractive(text, model, model_name, tokenizer):
    device = "cuda:0" 
    model = model.to(device)
    max_length = 512 if model_name == "pegasus-xsum" else 1024
    input_ids = tokenizer(
        [text],
        return_tensors = "pt",
        padding = "longest",       
        truncation = True,
        max_length = max_length 

    )["input_ids"].to(device)

    output_ids = model.generate(
        input_ids = input_ids
    )[0].to(device)

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens = True,
        clean_up_tokenization_spaces = False
    )
    return summary


def evaluation(language, lang_path, dataset): # Evaluation

    # Choose the proper language for the path setup.
    language = 'english' if lang_path == en_path else 'greek'
    dirpath = os.path.join(dataset_path, lang_path, "produced")
    sumpath = os.path.join(dataset_path, lang_path, "summary")

    scorers = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum','bleu-1', 'bleu-2', 'sacrebleu'] 
    
    #Init Greek Rouge and gr_tokenizer
    gr_tokenizer = Tokenizer('greek')._get_word_tokenizer('greek')
    gr_rouge = rouge_scorer.RougeScorer(scorers[:4], use_stemmer = False, tokenizer = gr_tokenizer )

    # Make a List of text files in the wanted file directory.
    method_directories = next(os.walk(dirpath))[1][0:]
    pathlib.Path(results_path).mkdir(parents = True, exist_ok = True)
    
    sum_score = 0 
    for scorer in scorers:
        data = {dataset: []}
        print(scorer + ' start')
        for method in method_directories:
            files = os.listdir(os.path.join(dirpath, method))
            method_score = 0 
            print(method)
            for file in files:
                print(file)
                with open(os.path.join(dirpath, method, file), 'r', encoding = 'utf-8-sig', errors = 'ignore') as h, \
                    open(os.path.join(sumpath, file.replace("baseline", "summary")), 'r', encoding = 'utf-8-sig', errors = 'ignore') as r:
                    hyp = h.read() if language == 'english' else prepare(h.read())
                    ref = r.read() if language == 'english' else prepare(r.read())
                    if hyp and not hyp.isspace():
                        method_score += score(hyp, ref, scorer, language, gr_rouge)

            data[dataset].append(method_score/len(files))

        # Construct the dataframe.    
        df = pd.DataFrame(data = data)
        df.index = method_directories

        # Set the index to the first column and save to excel.
        df.to_excel(os.path.join(results_path, f"{dataset}_{language}_{scorer}_output.xlsx"))

    return
