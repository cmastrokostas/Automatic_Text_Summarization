from SumSurvey.summarization import summarization, evaluation, macro_score
from SumSurvey.config import en_path, el_path, multiling_path

def main():

    languages = {'english': en_path, 'greek': el_path}
    for language in languages: 
        #summarization(language, languages[language])
        #evaluation(language, languages[language])
        macro_score(language, languages[language], multiling_path)
    
    return

if __name__ == '__main__': main()