from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from datasets import load_metric

# Set each time,to get results for the desired dataset. For example 'D:\datasets\cnn_dailymail'
dataset_path = r'C:\Users\xary_\source\repos\Unsupervised_Text_Summarization_Survey\MultiLingPilot2013'

# Set where you want your results to be saved.
results_path = r'C:\Users\xary_\source\repos\Unsupervised_Text_Summarization_Survey\Results' 

baseline_path = 'baseline' # Original Text Directory.
summary_path = 'summary' # Human Generated Summaries.

el_path = 'el'
en_path = 'en'
 
debug = False
n_sentences = 3

sumy_summarizers = { 
    'LexRank': LexRankSummarizer(),
    'TextRank': TextRankSummarizer(),
    'Luhn': LuhnSummarizer(),
    'Lsa': LsaSummarizer(),
}

pytextrank_summarizers = {
    'pyTextRank': 'textrank',
    'PositionRank': 'positionrank',
    'BiasedRank': 'biasedtextrank',
    'TopicRank': 'topicrank'
}

huggingface_metrics = {
    "rouge": load_metric("rouge"),
    "sacrebleu": load_metric("sacrebleu"),
    "bleu": load_metric('bleu'),
} 

abstractive_models = {
    "Distilbart-cnn-12-6": "sshleifer/distilbart-cnn-12-6",
    "Distilbart-xsum-12-6":"sshleifer/distilbart-xsum-12-6",
    "mT5-multilingual-XLSum": "csebuetnlp/mT5_multilingual_XLSum",
    "bart-large-cnn": "facebook/bart-large-cnn",
    "bart-large-xsum": "facebook/bart-large-xsum",
    "pegasus-large": "google/pegasus-large",
    "pegasus-multi_news": "google/pegasus-multi_news",
    "pegasus-xsum": "google/pegasus-xsum"
}

greek_abstractive_models = {
    "mt5_greek_translated_XLSum": "greek_models/mt5_greek_translated_XLSum",
}
