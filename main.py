from SumSurvey.summarization import summarization, evaluation
from SumSurvey.config import en_path, el_path, dataset_path


def main():

    languages = {'greek': el_path, 'english': en_path}

    for language in languages:  
        summarization(language, languages[language])
        evaluation(language, languages[language], dataset_path)
    return

if __name__ == '__main__': main()