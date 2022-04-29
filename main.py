from SumSurvey.summarization import summarization
from SumSurvey.config import en_path, el_path

def main():

    languages= {1: el_path, 2: en_path}
    for lang in languages : 
        summarization(languages[lang])
    #while True:
    #    inp = int(input("Choose '1' for Greek or '2' for English texts : "))
    #    if (inp == 1 or inp == 2 ):
    #        print(summarization(languages[inp]))
    #        break
    #    else :
    #        print("Try Again...")
    print("Done...")
    return

if __name__ == '__main__': main()