from SumSurvey.summarization import summarization
from SumSurvey.config import en_path, el_path



def main():

    languages= {1: el_path, 2: en_path}

    while True:
        inp = int(input("Choose '1' for Greek or '2' for English texts : "))
        
    
        if (inp == 1 or inp == 2 ):

            print(summarization(languages[inp]))

            #print("Process Done Check File !")
            
            break
        else :
            print("Try Again...")

    return(0)

main()