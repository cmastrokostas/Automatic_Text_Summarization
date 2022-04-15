
import sumy
import pytextrank
import spacy 
import os
import string 
import json

from SumSurvey.config import multiling_path, output_path , body_path, el_path ,en_path, n_sentences


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer


from os.path import isfile # str join 

summarizers = {'LexRank': LexRankSummarizer(),'TextRank': TextRankSummarizer() ,'Luhn': LuhnSummarizer(),'Lsa': LsaSummarizer()}



def summarization(lang_path):

    if(lang_path == el_path):
        language = "greek"
    elif(lang_path == en_path):
        language = "english"

    path = str.join('',(multiling_path,body_path,lang_path))
    text_files=os.listdir(path)

    
    summaries = []
    i=0

    for text_file in text_files:
        dir = os.path.join(path,text_file)
        with open(dir,'r', encoding = 'utf-8-sig', errors = 'ignore') as file:
            parser = PlaintextParser.from_file(dir,Tokenizer(language))
            for summarizer in summarizers:  
                summary = summarizers[summarizer](parser.document,n_sentences)
                item = {f" {summarizer} Summarizer": summary}
                summaries.append(item)

    #with open ('json_summaries.json','w', encoding='utf8') as json_file:                       
    #    json.dump(summaries,json_file, ensure_ascii=False, indent = 4, separators = (',', ':'))
    
    # Seems to work but Cant use json.dump because of type <sentence> in created summary 
        return(summaries)


        





