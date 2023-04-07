# Dataset Translation
This subfolder contains code that translates a dataset from english to greek.

To translate the dataset, it should be prepared in the proper format as described  [here](https://github.com/cmastrokostas/Automatic_Text_Summarization/tree/main/Dataset%20Preparation). 

We used the Google Translate Ajax API through the [`googletrans`](https://pypi.org/project/googletrans/4.0.0rc1/) library in order to obtain a translated sample of 10.000 texts from the [XL-Sum dataset](https://github.com/csebuetnlp/xl-sum). This was done since we could not find a large greek summarization dataset. We used the translated texts in order to fine tune [`mt5-base`](https://huggingface.co/google/mt5-base) for greek text summarization.
## Run 
Set the dataset directory path in `main.py` and run.
