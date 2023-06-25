import os 
import json
import re



files = ['test.json', 'train.json', 'val.json']

# Merge json files for SAMSum
def merge_JsonFiles(filename):
    result = list()
    for f1 in filename:
        with open(f1, 'r', encoding = 'utf-8-sig', errors='ignore') as infile:
            result.extend(json.load(infile))

    with open('complete.json', 'w',encoding = 'utf-8-sig', errors='ignore') as output_file:
        json.dump(result, output_file)

merge_JsonFiles(files)

# Process for SAMSum
with open("complete.json", 'r', encoding = 'utf-8-sig', errors='ignore') as json_file :
    json_list = json.load(json_file)

    base_sentence_count = 0
    sum_sentence_count = 0
    
    base_tokens_count = 0
    sum_tokens_count = 0

for  json_str in json_list:
    instance =  json_str

    base_sentence_count += len(re.split(r'[.!?$\n]+', instance['dialogue']))
    sum_sentence_count += len(re.split(r'[.!?$\n]+', instance['summary']))

    base_tokens_count += len(instance['dialogue'].split())
    sum_tokens_count += len(instance['summary'].split())

print(f"SAMSum: \nBaseline mean sentences: {base_sentence_count/len(json_list)}, Summaries mean sentences: {sum_sentence_count/len(json_list)} \n\
Baseline mean tokens: {base_tokens_count/len(json_list)}, Summaries mean sentences: {sum_tokens_count/len(json_list)}")

files = ['english_test.jsonl', 'english_train.jsonl', 'english_val.jsonl']

# Merge jsonl files
def merge_jsonfiles(filename):

    outfile = open('complete_xlsum.jsonl','w', encoding= 'utf-8-sig')
    for f in filename:
        with open(f, 'r', encoding='utf-8-sig') as infile:
            for line in infile.readlines():
                outfile.write(line)
    outfile.close()

merge_jsonfiles(files)


# Process for XL-Sum

with open("complete_xlsum.jsonl", 'r', encoding = 'utf-8-sig', errors='ignore') as json_file :
    json_list = list(json_file)

    base_sentence_count = 0
    sum_sentence_count = 0
    
    base_tokens_count = 0
    sum_tokens_count = 0

for  json_str in json_list:
    instance =  json.loads(json_str)

    base_sentence_count += len(re.split(r'[.!?&\n]+', instance['text'][0:-1]))
    sum_sentence_count += len(re.split(r'[.!?\n]+', instance['summary'][0:-1]))
   
    base_tokens_count += len(instance['text'].split())
    sum_tokens_count += len(instance['summary'].split())

print(f"XL-Sum: \nBaseline mean sentences: {base_sentence_count/len(json_list)}, Summaries mean sentences: {sum_sentence_count/len(json_list)} \n\
Baseline mean tokens: {base_tokens_count/len(json_list)}, Summaries mean tokens: {sum_tokens_count/len(json_list)}")