import sumy
import pytextrank
import os
import json

from SumSurvey.config import multiling_path, output_path, body_path, el_path, en_path, n_sentences
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

summarizers = { 'LexRank': LexRankSummarizer(), 'TextRank': TextRankSummarizer(), 'Luhn': LuhnSummarizer(), 'Lsa': LsaSummarizer() }

def summarization(lang_path):
    # Choose the proper language for the path setup.
    language = 'english' if lang_path == en_path else 'greek'

    # Set proper file directory path.
    path = os.path.join(multiling_path, body_path, lang_path)
    
    # Make a List of text files in the wanted file directory.
    text_files = os.listdir(path)

    # Initialize empty set of summaries for all files.
    sum_set=[]
    
    # Iterate through files in text files list. 
    for text_file in text_files:
        with open(os.path.join(path,text_file),'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
            # Initialize empty set of summaries for each file .
            items=[]
            # PlaintextParser converts the document in the proper form to be summarized.
            parser = PlaintextParser.from_file(os.path.join(path,text_file),Tokenizer(language))
            
            # Produce the summaries for each algorithm.
            for summarizer in summarizers:  
                summary = summarizers[summarizer](parser.document,n_sentences)
                sum_text = []
                
                # Keep only the text field of each sentence of the summary.
                for sent in summary:
                    sum_text.append(str(sent))
                item =  { f"{summarizer}" : f" {' '.join(sum_text)}" } 
                # Add each algorithm's summary in the set for each file.
                items.append(item)
            # Add each file's results in the set for all the files.
            sum_set.append({f"{text_file}":items})

    # Store the results in a json file.
    with open (f"{language}_json_summaries.json",'w', encoding = 'utf8') as json_file:                       
        json.dump(sum_set,json_file, ensure_ascii=False, indent = 4, separators = (',', ':'))
    return