import sumy
import pytextrank
import os
import json
import rouge

from rouge import Rouge
from SumSurvey.config import multiling_path, body_path, summary_path, el_path, en_path, n_sentences
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

summarizers = { 'LexRank': LexRankSummarizer(), 'TextRank': TextRankSummarizer(), 'Luhn': LuhnSummarizer(), 'Lsa': LsaSummarizer() }

def path_config(dataset_path,type_path,lang_path):
    # Set proper file directory path.
    path = os.path.join(dataset_path, type_path, lang_path)
    return path

def summarization(lang_path):
    # Choose the proper language for the path setup.
    language = 'english' if lang_path == en_path else 'greek'

    path = path_config(multiling_path, body_path, lang_path)

    # Make a List of text files in the wanted file directory.
    text_files = os.listdir(path)
    

    # Initialize empty set of summaries for all files.
    files_set=[]
    
    # Iterate through files in text files list. 
    for text_file in text_files:
        with open(os.path.join(path,text_file),'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
            # Initialize empty set of summaries for each file .
            items = []
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
            files_set.append({f"{text_file}":items})

    # Store the results in a new file.
    with open (f"{language}_json_summaries.json",'w', encoding = 'utf8') as json_file:                       
        json.dump(files_set,json_file, ensure_ascii=False, indent = 4, separators = (',', ':'))
    return

def evaluation(lang_path):
    # Choose the proper language for the path setup.
    language = 'english' if lang_path == en_path else 'greek'
    hyp_path = os.path.join(r'C:\Users\Charalampos\source\repos\Unsupervised_Text_Summarization_Survey',f"{language}_json_summaries.json")
    ref_path = path_config(multiling_path, summary_path, lang_path)

    # Make a List of text files in the wanted file directory.
    summary_files = os.listdir(ref_path)

    # Evaluation
    rouge = Rouge()

    with open(hyp_path,'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
        data = json.load(file)
    i = 0 
    files_set = []
    
    # Iterate through summary files. 
    for summary_file in summary_files : 
        file_set = []
        with open (os.path.join(ref_path,summary_file),'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
            refs = file.read()
        hyp_name = summary_file.replace(str("_summary"),"_body") # Format name properly.
        j = 0

        # Iterate through summarizers to score each algorithm.
        for summarizer in summarizers:
            hyps = data[i][hyp_name][j][summarizer]
            scores = rouge.get_scores(hyps, refs)
            # Add each algorithm's results in the set for each file.
            item = { f"{summarizer}" : scores }
            file_set.append(item)
            j += 1

        # Add each file's results in the set for all the files.
        items = {summary_file.replace(str("_summary.txt"),"") : file_set }
        files_set.append(items)
        i += 1

    # Store the results in a new file.
    with open (f"{language}_scores.json",'w', encoding = 'utf8') as json_file:                       
        json.dump(files_set,json_file, ensure_ascii=False, indent = 4, separators = (',', ':'))
    return 