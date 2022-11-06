from functions import billsum_prep, xsum_prep, cnn_prep, samsum_prep, xlsum_prep, reddit_tifu_prep


def main():
    
    #Folder containing the required original dataset files or directories.
    original_folder = r'C:\Users\xary_\source\repos\Unsupervised_Text_Summarization_Survey\Dataset Preparation\Original Datasets'
    destination_folder = r'C:\Users\xary_\source\repos\Unsupervised_Text_Summarization_Survey\Dataset Preparation\Evaluation Datasets'

    billsum_prep(original_folder, destination_folder)
    xsum_prep(original_folder, destination_folder)
    cnn_prep(original_folder, destination_folder)
    samsum_prep(original_folder, destination_folder)
    xlsum_prep(original_folder, destination_folder)
    reddit_tifu_prep(original_folder, destination_folder)

    return


if __name__ == '__main__': main()