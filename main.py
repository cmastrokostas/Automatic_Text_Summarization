from SumSurvey.summarization import summarization, evaluation, pyTextRank
from SumSurvey.config import en_path, el_path
from SumSurvey.abstractive_tests import test
from datasets import load_metric
from rouge_score import rouge_scorer
import torch
import sys


def main():

    languages = {'greek': el_path, 'english': en_path}
    for language in languages: 
        summarization(language, languages[language])
        evaluation(language, languages[language])
    return

if __name__ == '__main__': main()