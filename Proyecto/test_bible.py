# Library imports

import os
import pandas as pd
from nltk import sent_tokenize
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import string, unicodedata, re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pickle

nltk.download('punkt')
nltk.download('wordnet')
import os

directory_path_sp = './/content//biblia_espanol//'

all_file_text_spanish = []

# Function to extract the numerical part of a filename
def extract_file_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return 0

# Get a list of files in the directory and sort them by their numerical part
files_in_directory_sp = os.listdir(directory_path_sp)
files_in_directory_sp = sorted(files_in_directory_sp, key=lambda x: extract_file_number(x))

for filename in files_in_directory_sp:
    if filename.startswith('biblia_espanol_') and filename.endswith('.txt'):
        file_path = os.path.join(directory_path_sp, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            start_index = text.find("\n1")
            file_text = text[start_index + len("\n1"):]
            file_text = file_text.replace("\n", " ")
            file_text = ' '.join(file_text.split())
            file_text = re.sub(r'[^\w\s]', '', file_text)
            file_text = unicodedata.normalize('NFKD', file_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            file_text = re.sub(r'(\d+)', r'.', file_text)
            file_text = re.sub(r' (?=\.)', '', file_text)
            file_text = file_text.replace("..", ".")
            file_text = re.sub(r'(\.)', r'. ', file_text)
            file_text = file_text.split('. ')
            file_text = [text for text in file_text if text != ""]
            all_file_text_spanish += file_text

directory_path_wa = './/content//biblia_wayuu/'

all_file_text_wayuu = []


# Get a list of files in the directory and sort them by their numerical part
files_in_directory_wa = os.listdir(directory_path_wa)
files_in_directory_wa = sorted(files_in_directory_wa, key=lambda x: extract_file_number(x))

for filename in files_in_directory_wa:
    if filename.startswith('biblia_wayuu_') and filename.endswith('.txt'):
        file_path = os.path.join(directory_path_wa, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            start_index = text.find("\n1")
            file_text = text[start_index + len("\n1"):]
            file_text = file_text.replace("\n", " ")
            file_text = ' '.join(file_text.split())
            file_text = re.sub(r'[^\w\s]', '', file_text)
            file_text = unicodedata.normalize('NFKD', file_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            file_text = re.sub(r'(\d+)', r'.', file_text)
            file_text = re.sub(r' (?=\.)', '', file_text)
            file_text = file_text.replace("..", ".")
            file_text = re.sub(r'(\.)', r'. ', file_text)
            file_text = file_text.split('. ')
            file_text = [text for text in file_text if text != ""]
            all_file_text_wayuu += file_text


print(f"wayuu lines: {len(all_file_text_wayuu)}, spanish lines:{len(all_file_text_spanish)}")