import os 
import json
import pathlib
import csv 
from csv import DictReader 
import random


def billsum_prep(original_folder, destination_folder):
    dataset = os.path.join(original_folder, 'us_test_data_final_OFFICIAL.jsonl')
    pathlib.Path(os.path.join(destination_folder, 'BillSum', 'en', 'baseline')).mkdir(parents = True, exist_ok = True)
    pathlib.Path(os.path.join(destination_folder, 'BillSum', 'en', 'summary')).mkdir(parents = True, exist_ok = True)
    
    with open(dataset, 'r', encoding = 'utf-8-sig', errors ='ignore') as json_file :
        json_list = list(json_file)

    for  json_str in json_list:
        instance = json.loads(json_str)
        with open(os.path.join(destination_folder, 'BillSum', 'en', 'baseline', f"{instance['bill_id']}.txt"), 'w', encoding = 'utf-8-sig', errors ='ignore') as base, \
             open(os.path.join(destination_folder, 'BillSum', 'en', 'summary', f"{instance['bill_id']}.txt"), 'w', encoding = 'utf-8-sig', errors = 'ignore') as summary:
                base.write(instance['text'])
                summary.write(instance['summary'])
    return


def xsum_prep(original_folder, destination_folder):
    # Open file containing the dataset split.
    with open(os.path.join(original_folder, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json'), 'r', encoding = 'utf-8') as split:
        data = json.load(split)

    # List of test split file ids.
    test = data['test']
    
    
    pathlib.Path(os.path.join(destination_folder, 'xsum', 'en', 'baseline')).mkdir(parents = True, exist_ok = True)
    pathlib.Path(os.path.join(destination_folder, 'xsum', 'en', 'summary')).mkdir(parents = True, exist_ok = True)
    
    # Make sure the .summary files from the original dataset are saved in the proper directory.
    files_origin = os.path.join(original_folder, 'xsum', 'en', 'raw')

    dir_list = os.listdir(files_origin)
    print(dir_list)

    for filename in dir_list:
        f = os.path.join(files_origin, filename)
    
        if filename.replace('.summary', '') in test:
            filename = filename.replace('summary', 'txt')
            with open(f, encoding = "utf-8-sig") as file:
                with open (os.path.join(destination_folder, 'xsum', 'en', 'summary', filename), 'w', encoding = "utf-8-sig") as new, \
                     open (os.path.join(destination_folder, 'xsum', 'en', 'baseline', filename), 'w', encoding = "utf-8-sig") as base:
                    copy = False
                    flag = False
                    for line in file:
                        if line.strip() == "[SN]FIRST-SENTENCE[SN]":
                            copy = True
                            continue
                        elif line.strip() == "[SN]RESTBODY[SN]":
                            copy = False
                            flag = True
                            continue
                        elif copy:
                            new.write(line)
                        elif flag:
                            base.write(line)
    return


def cnn_prep(original_folder, destination_folder):
    
    pathlib.Path(os.path.join(destination_folder, 'cnn_dailymail', 'en', 'baseline')).mkdir(parents = True, exist_ok = True)
    pathlib.Path(os.path.join(destination_folder, 'cnn_dailymail', 'en', 'summary')).mkdir(parents = True, exist_ok = True)

    with open(os.path.join(original_folder, 'cnn_test.csv'), encoding = "utf-8-sig") as read_obj:
        csv_dict_reader = DictReader(read_obj)
        
        for row in csv_dict_reader:
            txt_file = f"{row['id']}.txt"
            with open(os.path.join(destination_folder, 'cnn_dailymail', 'en', 'baseline', txt_file), "w", encoding = "utf-8-sig") as baseline, \
                 open(os.path.join(destination_folder, 'cnn_dailymail', 'en', 'summary', txt_file), "w", encoding = "utf-8-sig") as summary:
                baseline.write(row['article'])
                summary.write(row['highlights'])
    return


def samsum_prep(original_folder, destination_folder):
    pathlib.Path(os.path.join(destination_folder, 'SamSum', 'en', 'baseline')).mkdir(parents = True, exist_ok = True)
    pathlib.Path(os.path.join(destination_folder, 'SamSum', 'en', 'summary')).mkdir(parents = True, exist_ok = True)

    with open(os.path.join(original_folder, "samsum_test.json"), 'r', encoding = 'utf-8-sig', errors = 'ignore') as json_file :
        json_list = json.load(json_file)

    for  json_str in json_list['file']:
        instance = json_str

        with open(os.path.join(destination_folder, 'SamSum', 'en', 'baseline', f"{instance['id']}.txt"), 'w', encoding = 'utf-8-sig', errors = 'ignore') as base, \
             open(os.path.join(destination_folder, 'SamSum', 'en', 'summary', f"{instance['id']}.txt"), 'w', encoding = 'utf-8-sig', errors = 'ignore') as summary:
            base.write(instance['dialogue'])
            summary.write(instance['summary'])

    return


def xlsum_prep(original_folder, destination_folder):

    pathlib.Path(os.path.join(destination_folder, 'XLSum', 'en', 'baseline')).mkdir(parents = True, exist_ok = True)
    pathlib.Path(os.path.join(destination_folder, 'XLSum', 'en', 'summary')).mkdir(parents = True, exist_ok = True)

    with open(os.path.join(original_folder, "english_test.jsonl"), 'r', encoding = 'utf-8-sig', errors = 'ignore') as json_file :
        json_list = list(json_file)
    for json_str in json_list:
        instance = json.loads(json_str)
        with open(os.path.join(destination_folder, 'XLSum', 'en', 'baseline', f"{instance['id']}.txt"), 'w', encoding = 'utf-8-sig', errors='ignore') as base, \
             open(os.path.join(destination_folder, 'XLSum', 'en', 'summary', f"{instance['id']}.txt"), 'w', encoding = 'utf-8-sig', errors='ignore') as summary:
            base.write(instance['text'])
            summary.write(instance['summary'])
    return


def reddit_tifu_prep(original_folder, destination_folder):

    pathlib.Path(os.path.join(destination_folder, 'RedditTIFU', 'en', 'baseline')).mkdir(parents = True, exist_ok = True)
    pathlib.Path(os.path.join(destination_folder, 'RedditTIFU', 'en', 'summary')).mkdir(parents = True, exist_ok = True)

    # Extract all the files from the original json to baseline and summary text files.
    with open(os.path.join(original_folder, "tifu_all_tokenized_and_filtered.json"), 'r', encoding = 'utf-8-sig', errors ='ignore') as json_file:
        json_list = list(json_file)
    valid_file_list = []
    for  json_str in json_list:
        instance = json.loads(json_str)
        
        if instance['tldr'] and any(c.isalnum() for c in instance['tldr']):
            valid_file_list.append(instance['id'])
    random.shuffle(valid_file_list) 
    test_data = valid_file_list[:int(0.05*len(valid_file_list))]

    for i in json_list:
        instance = json.loads(i)
        if instance['id'] in test_data:
            with open(os.path.join(destination_folder, 'RedditTIFU', 'en', 'baseline', f"{instance['id']}.txt"), 'w', encoding = 'utf-8-sig', errors = 'ignore') as baseline, \
                 open(os.path.join(destination_folder, 'RedditTIFU', 'en', 'summary', f"{instance['id']}.txt"), 'w', encoding = 'utf-8-sig', errors = 'ignore') as summary:
                    baseline.write(instance['selftext_without_tldr'])
                    summary.write(instance['tldr'])
    return
