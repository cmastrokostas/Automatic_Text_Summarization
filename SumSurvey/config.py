from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

multiling_path = r'C:\Users\Charalampos\source\repos\Unsupervised_Text_Summarization_Survey\datasets\MultiLingPilot2013'
results_path = r'C:\Users\Charalampos\source\repos\Unsupervised_Text_Summarization_Survey\Results'
body_path = r'body\text'
baseline_path = 'baseline'
summary_path = 'summary'
el_path = 'el'
en_path = 'en'

summaries_file = "json_summaries.json"
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
    'PositionRank' : 'positionrank',
    'BiasedRank' : 'biasedtextrank',
    'TopicRank' : 'topicrank'
}