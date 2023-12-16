# IMPORTANT: run this file from the project root directly, like "python lda/lda.py"

import string
import pandas as pd
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


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

def perform_lda(dataframe, num_topics=10):
    # Preprocessing the transcript data
    doc_clean = [preprocess(doc).split() for doc in dataframe['transcript']]  

    # Creating the term dictionary of our corpus, where every unique term is assigned an index
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Creating the LDA model
    lda_model = models.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=50)

    return lda_model, dictionary


## HYPERPARAMETERS
num_topics = 10

# Load the data
df = pd.read_csv("video_transcripts_with_hierarchy_mapped_truncated_1702443584.csv")
print("Dataframe head:")
print(df.head())
print("----------------------------------")

# Performing LDA
lda_model, term_dict = perform_lda(df, num_topics=num_topics)

# Display the topics
print("Top words associated with each topic:")
topics = lda_model.print_topics(num_topics=10, num_words=5)
for topic in topics:
    print(topic)

#### We've trained our LDA model. Now, let's get the topic distributions for each video transcript ####

for i in range(num_topics):
    df[f'topic_{i}'] = 0.0

# Function to get the topic distribution for a document
def get_topic_distribution(lda_model, bow):
    topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0)
    return dict(topic_distribution)

# Updating the DataFrame with topic weightings
for idx, row in df.iterrows():
    # Preprocess the transcript
    processed_transcript = preprocess(row['transcript'])
    bow = term_dict.doc2bow(processed_transcript.split())

    # Get the topic distribution
    topic_dist = get_topic_distribution(lda_model, bow)

    # Update the DataFrame with topic weights
    for topic_id, weight in topic_dist.items():
        df.at[idx, f'topic_{topic_id}'] = weight

# Now df has additional columns for each topic's weighting
print("DataFrame with topic weightings:")
print(df.head())

# Save the csv with topic distributions, the LDA model, and the term dictionary: 
df.to_csv("video_transcripts_with_hierarchy_mapped_truncated_1702443584.csv", index=False)
lda_model.save('lda_model.model')
term_dict.save('term_dict.dict')
