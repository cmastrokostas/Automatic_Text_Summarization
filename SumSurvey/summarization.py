import sumy
import pytextrank
import os
import json
import rouge
import spacy 


from rouge import Rouge
from SumSurvey.config import multiling_path, baseline_path, summary_path, results_path, en_path, summaries_file
from SumSurvey.config import n_sentences, sumy_summarizers, pytextrank_summarizers, huggingface_metrics
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from datasets import load_metric


def summarization(language, lang_path):

    path = os.path.join(multiling_path, lang_path, baseline_path)

    # Make a List of text files in the wanted file directory.
    text_files = os.listdir(path)

    # Initialize empty set of summaries for all files.
    files_set = []
    
    # Iterate through files in text files list. 
    for text_file in text_files:

        # PlaintextParser converts the document in the proper form to be summarized.
        parser = PlaintextParser.from_file(os.path.join(path, text_file), Tokenizer(language))
        
        # Produce summaries for sumy algorithms.
        sumy_summaries = {
            summarizer: ' '.join([
                str(sent) for sent in sumy_summarizers[summarizer](parser.document, n_sentences)
            ]) for summarizer in sumy_summarizers
        }

        with open(os.path.join(path, text_file), 'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
            file1 = file.read().replace("\n"," ")  # file_text = ' '.join([str(sentence) for sentence in parser.document.sentences])

        # Produce summaries for pyTextRank algorithms.
        spacy_summaries = {
            summarizer: (pyTextRank(file1, pytextrank_summarizers[summarizer], language)) for summarizer in pytextrank_summarizers
        }

        # Join Dictionaries
        summaries = sumy_summaries|spacy_summaries

        # Add each file's results in the set for all the files.
        files_set.append({text_file: summaries})

    # Store the results in a new file.
    with open (os.path.join(results_path, f"{language}_{summaries_file}"), 'w', encoding = 'utf-8-sig', errors = 'ignore') as json_file:                       
        json.dump(files_set, json_file, ensure_ascii = False, indent = 4, separators = (',', ': '))
    return


def evaluation(language, lang_path):

    # Choose the proper language for the path setup.
    language = 'english' if lang_path == en_path else 'greek'
    
    # Hypothesis path (machine generated) & Reference path (author assigned) summaries. 
    hyp_path = os.path.join(results_path, f"{language}_{summaries_file}")
    ref_path = os.path.join(multiling_path, lang_path, summary_path)

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

        hyp_file = summary_file.replace("_summary", "_baseline")
        
        file_set = {
            summarizer: {rouge.get_scores(str(hypothesis[hyp_file][summarizer]), reference[0])[0] |\
            {hug: huggingface_metrics[hug].compute(predictions = [hypothesis[hyp_file][summarizer]], references = [reference]) for hug in huggingface_metrics}
            }for summarizer in sumy_summarizers | pytextrank_summarizers
        }
            

        # Add each file's results in the set for all the files.
        files_set.append({summary_file.replace("_summary.txt","_body.txt"): file_set})

    # Store the results in a new file.
    with open (os.path.join(results_path, f"{language}_scores.json"),'w', encoding = 'utf8') as json_file:                       
        json.dump(files_set,json_file, ensure_ascii = False, indent = 4, separators = (',', ':'))
    return


def pyTextRank(text, summarizer, language):

    # Choose proper language model
    nlp = spacy.load("el_core_news_sm") if language == "greek" else spacy.load("en_core_web_sm")

    # Summarization pipeline
    nlp.add_pipe(factory_name = summarizer, name = summarizer, last = True)
    doc = nlp(text)
    summary = ' '.join([str(sent) for sent in doc._.textrank.summary(limit_sentences = n_sentences)]) # Join sentences

    return summary