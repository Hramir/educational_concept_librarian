# IMPORTANT: run this file from the project root directly, like "python lda/lda_test.py"

import string
import pandas as pd
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from utils.transcript_df import process_directory

# Ensure you have the NLTK data downloaded
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([word for word in text.lower().split() if word not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def perform_lda(dataframe):
    # Preprocessing the transcript data
    doc_clean = [preprocess(doc).split() for doc in dataframe['transcript']]  

    # Creating the term dictionary of our corpus, where every unique term is assigned an index
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Creating the LDA model
    lda_model = models.LdaModel(doc_term_matrix, num_topics=10, id2word=dictionary, passes=50)

    return lda_model

# Load the data
directory_path = 'data'
df = process_directory(directory_path)
print("Dataframe head:")
print(df.head())
print("----------------------------------")

# Performing LDA
lda_model = perform_lda(df)

# Display the topics
print("Top words associated with each topic:")
topics = lda_model.print_topics(num_topics=10, num_words=5)
for topic in topics:
    print(topic)
