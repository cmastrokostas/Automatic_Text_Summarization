from datasets import load_metric
import json
import string
import unicodedata

from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from env.rouge import rouge_scorer

def test():
    hyp_file = r"C:\Users\Charalampos\source\repos\Unsupervised_Text_Summarization_Survey\Results\greek_json_summaries.json"

    with open(hyp_file, 'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
        hypotheses_list = json.load(file)

    with open(r"C:\Users\Charalampos\source\repos\Unsupervised_Text_Summarization_Survey\my_datasets\MultiLingPilot2013\el\summary\0d86383293996518eee416037c5ff576_summary.txt", encoding = 'utf-8-sig') as file:
        reference = file.read()

    hypothesis = hypotheses_list[0]["0d86383293996518eee416037c5ff576_baseline.txt"]["TextRank"]

    clean_reference = reference.upper().translate(str.maketrans('', '', string.punctuation))
    clean_hypothesis = hypothesis.upper().translate(str.maketrans('', '', string.punctuation))

    stemmer = Stemmer("greek") 

    gr_tokenizer = Tokenizer('greek')._get_word_tokenizer('greek')

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer = False, tokenizer = gr_tokenizer )

    gr_rouge = scorer.score(prepare_greek_text(hypothesis), prepare_greek_text(reference))
    #original_rouge = scorer.score(hypothesis, reference)

    print("\n\n")
    print(f"Rouge with preprocessing {clean_rouge}")
    print("\n\n")
    #print(f"Rouge without preprocessing {original_rouge}")
    
    print(prepare_greek_text(hypothesis))

    return

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) 
            if unicodedata.category(c) != 'Mn')

def prepare_greek_text(text, stemmer = Stemmer("greek")):

    return ' '.join([strip_accents(stemmer(i)) for i in text.upper().translate(str.maketrans('', '', string.punctuation)).split()])