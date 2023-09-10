import numpy as np
import os
import re
import random

n_grams_folder = "n_grams"
group_code = "group_code" 

def custom_tokenizer(text):
    # Use regular expression to split on spaces while preserving '<s>' and '</s>' as separate tokens
    tokens = re.split(r'(\s|<s>|</s>)', text)
    
    # Remove empty tokens and tokens containing only special characters
    tokens = [token for token in tokens if token.strip() and not re.match(r'^[^a-zA-Z0-9]+$', token)]
    
    # Remove trailing '>' characters from tokens like "the>"
    tokens = [re.sub(r'(.*[^>])>', r'\1', token) if token not in ['<s>','</s>', '<UNK>'] else token for token in tokens]
    
    return tokens

# Load the n-gram model
def load_ngram_model(model_file):
    ngram_model = {}
    with open(model_file, "r", encoding="utf-8") as file:
        for line in file:
            ngram, probability = line.strip().split("\t")
            ngram_model[ngram] = float(probability)
    return ngram_model

# Function to generate the next word based on the n-gram model
def generate_next_word(ngram_model, current_words):
    candidates = []
    probabilities = []
    for key in ngram_model.keys():
        if key.startswith(current_words):
            candidates.append(key.split(" ")[-1])
            probabilities.append(ngram_model.get(key))
    if not candidates:
        # rwg = random.choice(list(ngram_model.keys())) 
        return None # No candidates found, returns random word in vocab
    
    # Calculate probabilities for each candidate word
    # probabilities = [ngram_model.get(f"{current_words} {candidate}", 0.0) for candidate in candidates]

    # Normalize the probabilities to ensure they sum to 1
    total_probability = sum(probabilities)

    probabilities = [prob / total_probability for prob in probabilities]

    # Use the calculated and normalized probabilities to randomly select the next word
    next_word = np.random.choice(candidates, p=probabilities)

    return next_word

# Function to generate a sentence
def generate_sentence(start_word, ngram_model, n, max_length=50):
    sentence = [start_word.lower()]
    
    while len(sentence) < max_length:
        current_words = " ".join(sentence[-(n-1):])
        next_word = generate_next_word(ngram_model, current_words)
        if next_word is None:
            break
        
        sentence.append(next_word)
        
        if next_word == "</s>":
            break
    
    return " ".join(sentence)

# Choose the n-gram model to use (e.g., trigrams)
n_gram_model_file = os.path.join(n_grams_folder, f"BAC_{group_code}_trigrams.txt")
# Load the selected n-gram model
ngram_model = load_ngram_model(n_gram_model_file)

# Example usage:
start_word = "you"  # Replace with your desired start word
generated_sentence = generate_sentence(start_word, ngram_model, n=3)

print("Generated Sentence:")
print(generated_sentence)