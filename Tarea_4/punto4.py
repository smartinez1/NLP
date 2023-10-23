import re
import tensorflow as tf
import tensorflow_hub as hub
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Constants
group_code = "santiagomartinez_201533279_camilocastaneda_202314092"
segment_length = 200

# Function to preprocess text
def processing_text(texto):
    # Paso 1: Remover con un expresión regular carateres especiales (no palabras).
    processed_feature = re.sub(r'\W', ' ', str(texto))
    # Paso 2: Remover ocurrencias de caracteres individuales
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
    # Paso 3: Remover números (Ocurrencias muy esporádicas en nuestro dataset)
    processed_feature = re.sub(r'[0-9]+', ' ', processed_feature)
    # Paso 4: Simplificar espacios concecutivos a un único espacio entre palabras
    processed_feature = re.sub(' +', ' ', processed_feature)
    # Paso 5: Pasar todo el texto a minúsculas
    processed_feature = processed_feature.lower()

    return processed_feature

# Load and preprocess the text data
def load_and_preprocess_text_data(folder_path, segment_length):
    texts = []
    labels = []

    for author in authors:
        author_folder = os.path.join(folder_path, author)
        for filename in os.listdir(author_folder):
            with open(os.path.join(author_folder, filename), "r", encoding="utf-8") as file:
                text = file.read()
                # Process the text
                processed_text = processing_text(text)
                # Split text into segments if needed
                sequences = [processed_text[i:i+segment_length] for i in range(0, len(processed_text), segment_length)]
                texts.extend(sequences)
                labels.extend([author] * len(sequences))

    return texts, labels

# Read and preprocess the text data
def get_corpus(folder_path):
    texts = []
    invalid_txt = ['']
    
    for root, _, files in os.walk(folder_path):
        for file_ in files:
            with open(os.path.join(root, file_), "r", encoding="utf-8") as f:
                for line in f:
                    pre_processed_text = processing_text(line)
                    if pre_processed_text not in invalid_txt:
                        texts.append(pre_processed_text)
    
    return texts

# Get authors
authors = os.listdir("book_datasets")

# Initialize and fit the Keras tokenizer
tokenizer = Tokenizer()
corpus = get_corpus("book_datasets")
corpus = [sentence for sentence in corpus if sentence!=' ' and sentence!='' and len(sentence.split())>3]
tokenizer.fit_on_texts(corpus)
vocab_size = len(tokenizer.word_index) + 1
print("vocab size: " + str(vocab_size))
# Load and preprocess the text data
texts, labels = load_and_preprocess_text_data("book_datasets", segment_length)

# Encode labels and one-hot encode
label_encoder = LabelEncoder()
label_encoder.fit(authors)
labels_encoded = label_encoder.transform(labels)
labels_one_hot = to_categorical(labels_encoded, num_classes=len(authors))

# Debugging print statements
print("Number of texts:", len(texts))
print("Number of labels:", len(labels_one_hot))
# Split the dataset into training, validation, and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels_one_hot, test_size=0.2, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(test_texts, test_labels, test_size=0.5, random_state=42)

# Debugging print statements
print("Number of training texts:", len(train_texts))
print("Number of testing texts:", len(test_texts))

# Load the pre-trained Word2Vec model from TensorFlow Hub
embedding_model_url = "https://tfhub.dev/google/Wiki-words-500/2"
hub_layer = hub.KerasLayer(embedding_model_url, input_shape=[], dtype=tf.string, trainable=False)

# Define your feed forward neural network
model = tf.keras.Sequential([
    hub_layer,  # The Word2Vec embedding layer
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(authors), activation='softmax')
])

model.summary()
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Convert the text data into numpy arrays
train_texts = np.array(train_texts)
train_labels = np.array(train_labels)
val_texts = np.array(val_texts)
val_labels = np.array(val_labels)
test_texts = np.array(test_texts)
test_labels = np.array(test_labels)

# Train the model
history = model.fit(
    train_texts, train_labels,
    epochs=100,  # You can adjust the number of epochs
    batch_size=64,
    validation_data=(val_texts, val_labels)
)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_texts, test_labels)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy * 100:.2f}%')