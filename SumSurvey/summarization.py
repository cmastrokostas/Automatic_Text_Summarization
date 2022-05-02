import sumy
import pytextrank
import os
import json
import rouge

from rouge import Rouge
from SumSurvey.config import multiling_path, body_path, summary_path, results_path, en_path, n_sentences, summaries_file
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

summarizers = { 
    'LexRank': LexRankSummarizer(),
    'TextRank': TextRankSummarizer(),
    'Luhn': LuhnSummarizer(),
    'Lsa': LsaSummarizer(),
    #'pyTextRank': 'pyTextRank()'
}

spacy_objects = 0 
def pyTextRank(parser.document, n_sentences):
    # join sentences in str 
    




def summarization(lang_path):
    # Choose the proper language for the path setup.
    language = 'english' if lang_path == en_path else 'greek'

    path = os.path.join(multiling_path, body_path, lang_path)

    # Make a List of text files in the wanted file directory.
    text_files = os.listdir(path)
   
    # Initialize empty set of summaries for all files.
    files_set = []
    
    # Iterate through files in text files list. 
    for text_file in text_files:

        # PlaintextParser converts the document in the proper form to be summarized.
        parser = PlaintextParser.from_file(os.path.join(path, text_file), Tokenizer(language))
            
        # Produce summaries from each algorithm.
        summaries = {
            summarizer: ' '.join([
                str(sent) for sent in summarizers[summarizer](parser.document, n_sentences)
            ]) for summarizer in summarizers
        } 

        # Add each file's results in the set for all the files.
        files_set.append({text_file: summaries})

    # Store the results in a new file.
    with open (os.path.join(results_path, f"{language}_{summaries_file}"), 'w', encoding = 'utf-8-sig', errors = 'ignore') as json_file:                       
        json.dump(files_set, json_file, ensure_ascii = False, indent = 4, separators = (',', ': '))
    return


def evaluation(lang_path):

    # Choose the proper language for the path setup.
    language = 'english' if lang_path == en_path else 'greek'
    
    # Hypothesis path (machine generated) & Reference path (author assigned) summaries. 
    hyp_path = os.path.join(results_path, f"{language}_{summaries_file}")
    ref_path = os.path.join(multiling_path, summary_path, lang_path)

    # Make a List of text files in the wanted file directory.
    summary_files = os.listdir(ref_path)

    # Evaluation
    rouge = Rouge()

    with open(hyp_path, 'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
        hypotheses_list = json.load(file)
    
    files_set = []
    # Iterate through summary files. 
    for summary_file, hypothesis in zip(summary_files, hypotheses_list): 
        with open (os.path.join(ref_path, summary_file), 'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
            reference = file.read()

        hyp_file = summary_file.replace("_summary", "_body")
        file_set = {
            summarizer: rouge.get_scores(hypothesis[hyp_file][summarizer], reference)[0] 
            for summarizer in summarizers
        } 
        
        # Add each file's results in the set for all the files.
        files_set.append({summary_file.replace("_summary.txt","_body.txt"): file_set})

    # Store the results in a new file.
    with open (os.path.join(results_path, f"{language}_scores.json"),'w', encoding = 'utf8') as json_file:                       
        json.dump(files_set,json_file, ensure_ascii = False, indent = 4, separators = (',', ':'))
    return 