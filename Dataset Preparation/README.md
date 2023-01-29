# Dataset Preparation
#### This subfolder contains code to prepare the datasets in the proper directory format for the evaluation process. 

## Datasets 
The original datasets as well as some supporting files containing their splits required for the process can be found through the following links :

* CNN / Daily Mail: https://github.com/abisee/cnn-dailymail
* XSum: https://github.com/EdinburghNLP/XSum
* BillSum: https://github.com/FiscalNote/BillSum
* XLSum: https://github.com/csebuetnlp/xl-sum
* SamSum: https://metatext.io/datasets/samsum
* Reddit TIFU: https://github.com/ctr4si/MMN

## Required data
Make sure to have the following files saved in the 'Original Datasets' folder.

* `cnn_test.csv` (Original cnn_dailymail test split data.)
* `us_test_data_final_OFFICIAL.jsonl` (Original BillSum test split data.)
* `english_test.jsonl` (Original XLSum test split data)
* `tifu_all_tokenized_and_filtered.json` (Reddit TIFU original data)
* `samsum_test.json` (Original SamSum test split data)
* `XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json` (Original XSum training-dev-test split identifiers)

* Also make sure to extract all the  `.summary` files from XSum under the `Dataset Preparation\Original Datasets\xsum\en\raw` directory.

## Run
Configurate the `original_folder` and `destination_folder` variables in `Dataset Preparation\main.py` to match your local folders.
Run `Dataset Preparation\main.py`
