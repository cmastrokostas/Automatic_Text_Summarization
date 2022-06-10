from SumSurvey.summarization import summarization, evaluation
from SumSurvey.config import en_path, el_path

def main():

    languages = {'greek': el_path, 'english': en_path}
    for language in languages: 
        summarization(language, languages[language])
        #evaluation(language, languages[language])
    
    return

if __name__ == '__main__': main()