import numpy as np
import zipfile
import tarfile
import os
from collections import defaultdict
import nltk
from nltk.util import ngrams
from tqdm import tqdm 
import xml.etree.ElementTree as ET
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import random
from collections import Counter

# Specify the folder name
n_grams_folder = "n_grams"

# Check if the folder already exists
if not os.path.exists(n_grams_folder):
    # If it doesn't exist, create it
    os.mkdir(n_grams_folder)
    print(f"Folder '{n_grams_folder}' created successfully.")
else:
    print(f"Folder '{n_grams_folder}' already exists.")


# Define a function to calculate Laplace-smoothed probabilities for n-grams
def laplace_smoothing(ngram_counts, total_tokens, vocab_size, alpha=1.0):
    smoothed_probabilities = {}
    for ngram, count in ngram_counts.items():
        smoothed_prob = (count + alpha) / (total_tokens + alpha * vocab_size)
        smoothed_probabilities[ngram] = smoothed_prob

    return smoothed_probabilities

# Define a function to generate n-grams from a list of tokens
def generate_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    return ngrams

# Define a function to build and save N-gram models with Laplace smoothing
def build_and_save_ngram_model(sentences, n, output_file):
    breakpoint()
    # Tokenize the sentences into words
    tokens = " ".join(sentences).split()

    # Generate n-grams
    ngrams = generate_ngrams(tokens, n)

    # Calculate n-gram counts
    ngram_counts = Counter(ngrams)

    # Calculate vocabulary size
    vocab_size = len(set(tokens))

    # Calculate total number of tokens
    total_tokens = len(tokens)

    # Apply Laplace smoothing to calculate smoothed probabilities
    smoothed_probabilities = laplace_smoothing(ngram_counts, total_tokens, vocab_size)

    # Save the smoothed probabilities to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        for ngram, prob in smoothed_probabilities.items():
            ngram_str = " ".join(ngram)
            file.write(f"{ngram_str}\t{prob}\n")

# Download NLTK data for tokenization
nltk.download('punkt')

group_code = "group_code" 
# Paths to preprocessed training files
preprocessed_training_file_20N = f"20N_{group_code}_training.txt"
preprocessed_training_file_BAC = f"BAC_{group_code}_training.txt"

# Load preprocessed training data from the files
with open(preprocessed_training_file_20N, "r", encoding="utf-8") as file:
    train_sentences_20N = file.read()

with open(preprocessed_training_file_BAC, "r", encoding="utf-8") as file:
    train_sentences_BAC = file.read()

# Paths to training files
training_file_20N = f"20N_{group_code}_training.txt"
training_file_BAC = f"BAC_{group_code}_training.txt"

# Paths to output files
output_file_20N_unigrams = f"20N_{group_code}_unigrams.txt"
output_file_20N_bigrams = f"20N_{group_code}_bigrams.txt"
output_file_20N_trigrams = f"20N_{group_code}_trigrams.txt"
output_file_BAC_unigrams = f"BAC_{group_code}_unigrams.txt"
output_file_BAC_bigrams = f"BAC_{group_code}_bigrams.txt"
output_file_BAC_trigrams = f"BAC_{group_code}_trigrams.txt"

# Build and save N-gram models with Laplace smoothing
build_and_save_ngram_model(train_sentences_20N, 1, os.path.join(n_grams_folder,output_file_20N_unigrams))
print("1-gram models with Laplace smoothing generated and saved.")
build_and_save_ngram_model(train_sentences_20N, 2, os.path.join(n_grams_folder,output_file_20N_bigrams))
print("2-gram models with Laplace smoothing generated and saved.")
build_and_save_ngram_model(train_sentences_20N, 3, os.path.join(n_grams_folder,output_file_20N_trigrams))
print("3-gram models with Laplace smoothing generated and saved.")
build_and_save_ngram_model(train_sentences_BAC, 1, os.path.join(n_grams_folder,output_file_BAC_unigrams))
print("1-gram models with Laplace smoothing generated and saved.")
build_and_save_ngram_model(train_sentences_BAC, 2, os.path.join(n_grams_folder,output_file_BAC_bigrams))
print("2-gram models with Laplace smoothing generated and saved.")
build_and_save_ngram_model(train_sentences_BAC, 3, os.path.join(n_grams_folder,output_file_BAC_trigrams))
print("3-gram models with Laplace smoothing generated and saved.")

# Print confirmation message
print("N-gram models with Laplace smoothing generated and saved.")