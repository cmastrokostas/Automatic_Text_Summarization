from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

from datasets import load_metric

multiling_path = r'C:\Users\Charalampos\source\repos\Unsupervised_Text_Summarization_Survey\my_datasets\MultiLingPilot2013'
results_path = r'C:\Users\Charalampos\source\repos\Unsupervised_Text_Summarization_Survey\Results'

baseline_path = 'baseline' # Original Text Directory.
summary_path = 'summary' # Human Generated Summaries.

el_path = 'el'
en_path = 'en'

summaries_file = "json_summaries.json"
debug = False
n_sentences = 5

sumy_summarizers = { 
    'LexRank': LexRankSummarizer(),
    'TextRank': TextRankSummarizer(),
    'Luhn': LuhnSummarizer(),
    'Lsa': LsaSummarizer(),
    
}

pytextrank_summarizers = {
    'pyTextRank': 'textrank',
    'PositionRank' : 'positionrank',
    'BiasedRank' : 'biasedtextrank',
    'TopicRank' : 'topicrank'
}

huggingface_metrics = {
    "rouge-hf":load_metric("rouge"),
    "sacreblue": load_metric('sacrebleu'),
    "blue": load_metric('bleu'),
    "bleurt": load_metric('bleurt'),
    "ter": load_metric('ter')
}