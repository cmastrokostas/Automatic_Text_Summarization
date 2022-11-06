import unicodedata
import string
import spacy
import os 
import pytextrank

from SumSurvey.config import dataset_path, results_path, huggingface_metrics
from SumSurvey.config import n_sentences, sumy_summarizers, pytextrank_summarizers
from sumy.nlp.stemmers import Stemmer

greek_nlp = spacy.load("el_core_news_sm")
english_nlp = spacy.load("en_core_web_sm")


# Remove Accentation
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) 
            if unicodedata.category(c) != 'Mn')


# Greek Preprocessing
def prepare(text, stemmer = Stemmer("greek")):
    return ' '.join(strip_accents(stemmer(i)) for i in text.upper().translate(str.maketrans('', '', string.punctuation)).split())


def save_text(text, text_file, dataset, lang_path, summarizer): # -> utils
   
    # Saved summary directories organisation: ./<dataset>/<lang>/produced/<summarizer>/<file_name>
    with open(os.path.join(dataset, lang_path, "produced", summarizer, f"{text_file}"), 'w', encoding = 'utf-8-sig', errors = 'ignore') as f:
        f.write(text)
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


def score(hyp, ref, scorer, language, gr_rouge):

    greek_scorers = {
        'rouge1': lambda: gr_rouge.score(hyp, ref)["rouge1"].fmeasure,
        'rouge2': lambda: gr_rouge.score(hyp, ref)["rouge2"].fmeasure,
        'rougeL': lambda: gr_rouge.score(hyp, ref)["rougeL"].fmeasure,
        'rougeLsum': lambda: gr_rouge.score(hyp, ref)["rougeLsum"].fmeasure,
        'bleu-1': lambda: huggingface_metrics["bleu"].compute(predictions = [hyp.split()], references = [[ref.split()]], max_order = 1)['bleu'],
        'bleu-2': lambda: huggingface_metrics["bleu"].compute(predictions = [hyp.split()], references = [[ref.split()]], max_order = 2)['bleu'],
        'sacrebleu': lambda: huggingface_metrics["sacrebleu"].compute(predictions = [hyp], references = [[ref]])['score']
    }

    english_scorers = {
        'rouge1': (lambda : huggingface_metrics["rouge"].compute(predictions = [hyp], references = [ref])["rouge1"].mid.fmeasure),
        'rouge2': (lambda : huggingface_metrics["rouge"].compute(predictions = [hyp], references = [ref])["rouge2"].mid.fmeasure),
        'rougeL': (lambda : huggingface_metrics["rouge"].compute(predictions = [hyp], references = [ref])["rougeL"].mid.fmeasure),
        'rougeLsum': (lambda : huggingface_metrics["rouge"].compute(predictions = [hyp], references = [ref])["rougeLsum"].mid.fmeasure),
        'bleu-1': (lambda: huggingface_metrics["bleu"].compute(predictions = [(hyp).split()], references = [[ref.split()]], max_order = 1)['bleu']),
        'bleu-2': (lambda: huggingface_metrics["bleu"].compute(predictions = [(hyp).split()], references = [[ref.split()]], max_order = 2)['bleu']),
        'sacrebleu': (lambda : huggingface_metrics["sacrebleu"].compute(predictions = [hyp], references = [[ref]])['score'])
    }

    return greek_scorers[scorer]() if language == 'greek' else english_scorers[scorer]()
