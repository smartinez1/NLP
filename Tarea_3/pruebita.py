import numpy as np
import os
from tqdm import tqdm 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import re
import warnings

# Filter out FutureWarnings from scikit-learn
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define the path to the dataset and categories
data_path = "Datasets\\Multi Domain Sentiment\\processed_acl\\processed_acl"
categories = ["books", "dvd", "electronics", "kitchen"]

# Define a function to extract reviews and their labels with a specified limit
def extract_reviews_and_labels(category, sentiment, max_reviews=None):
    reviews = []
    labels = []
    folder_path = os.path.join(data_path, category)
    filename = f"{sentiment}.review"
    file_path = os.path.join(folder_path, filename)
    
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            # se tiene en cuenta la entrada, se pone en formato de texto limpio, repitiendo las palabras cuando en la estructura "str:num" num!=1
            line_clean = re.sub(r'(\w+):(\d+)', lambda x: (x.group(1) + ' ') * (int(x.group(2)) - 1) + x.group(1), line)
            line_clean = line_clean.strip()
            if line_clean:
                # Split the line into features and label parts
                parts = line_clean.split("#label#:")
                if len(parts) == 2:
                    review = parts[0].strip()
                    label = parts[1].strip()
                    reviews.append(review)
                    labels.append(label)
                    if max_reviews is not None and len(reviews) >= max_reviews:
                        break  # Stop extracting if the maximum number of reviews is reached
    return reviews, labels

# Feature representation strategies
vectorizers = {
    "tf": CountVectorizer(), 
    "tfidf": TfidfVectorizer(),
}

# Initialize results dataframe
results_df = pd.DataFrame(
    columns=["Category", "Algorithm", "Representation", "Precision", "Recall", "F1", "Accuracy"]
)

# Filter out all warnings
warnings.filterwarnings("ignore")

for category in categories:
    positive_reviews, positive_labels = extract_reviews_and_labels(category, "positive", max_reviews=None)
    negative_reviews, negative_labels = extract_reviews_and_labels(category, "negative", max_reviews=None)
    X_test, y_test = extract_reviews_and_labels(category, "unlabeled", max_reviews=None)
    X_train = positive_reviews + negative_reviews
    y_train = positive_labels + negative_labels

    # # Split the data into train and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    # Perform sentiment analysis for each category, algorithm, and feature representation

    for algorithm in ["NB", "LR"]:
        for representation, vectorizer in vectorizers.items():
            print(f"Processing Category: {category}, Algorithm: {algorithm}, Representation: {representation}")
            # Initialize the vectorizer
            X_train_vectorized = vectorizer.fit_transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)
            
            # Train the classifier
            if algorithm == "NB":
                classifier = MultinomialNB()
            elif algorithm == "LR":
                classifier = LogisticRegression(max_iter=500, random_state=13)  #se reduce numero maximo de iteraciones pq se demora muchísimo
                
            classifier.fit(X_train_vectorized, y_train)
            
            # Predict sentiment
            y_pred = classifier.predict(X_test_vectorized)
            
            # Calculate evaluation metrics
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results in the dataframe
            results_df = results_df.append(
                {
                    "Category": category,
                    "Algorithm": algorithm,
                    "Representation": representation,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "Accuracy": accuracy,
                },
                ignore_index=True,
            )
            if algorithm=='NB':
                print('_________________________')
                continue
            # Get the top N features (words) for each class (positive and negative)
            feature_names = vectorizer.get_feature_names()
            top_n = 10  # Adjust the number of top features to display

            coef = classifier.coef_[0]  # For binary classification, there's only one set of coefficients
            top_features_indices = np.argsort(coef)[-top_n:]
            top_features = [feature_names[i] for i in top_features_indices]
            print(f"Category: {category}")
            print(f"Top {top_n} features: {top_features}")
            print()

            print('_________________________')

# Display the results
print("Results:")
print(results_df)