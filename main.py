from SumSurvey.summarization import summarization,evaluation
from SumSurvey.config import en_path, el_path

def main():

    languages= {1: el_path, 2: en_path}

    for lang in languages : 
        summarization(languages[lang])
        evaluation(languages[lang])

    print("Done...")
    return

if __name__ == '__main__': main()