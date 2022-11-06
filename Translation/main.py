import os
import sys
import json 
import time
import random
import pathlib

from googletrans import Translator

def translate_to_greek(origin_path, target_dir):
    dir_list = os.listdir(os.path.join(origin_path, target_dir))

    t = 0 
    translator = Translator()
    starttime = time.time()

    pathlib.Path(os.path.join(origin_path, f'translated-{target_dir}')).mkdir(parents = True, exist_ok = True)

    for instance in dir_list: 

        if t > 999:
             time.sleep(3600.0 - ((time.time() - starttime)))
             t = 0
             starttime = time.time()

        t += 1
        print(t)
        print(instance)

        with open(os.path.join(origin_path, target_dir, instance), 'r', encoding = 'utf-8', errors = 'ignore') as f:
            text = f.read()
        time.sleep(random.uniform(0.05,0.3))
        translation = translator.translate(text, src = 'en', dest = 'el').text

        with open(os.path.join(origin_path, f'translated-{target_dir}', instance), 'w', encoding = 'utf-8', errors = 'ignore') as w:
            w.write(translation)
    return


def main():

    # Set the correct origin path.
    origin_path = r'C:\Users\xary_\source\repos\Translation\Translation\cnn_dailymail\en'

    for target_dir in ['baseline', 'summary']:
        translate_to_greek(origin_path, target_dir)

    return


if __name__ == '__main__': main()