import pandas as pd
import json
import string
from helpers import count_concepts, lda_preprocess, get_lda_topic_distribution
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


# No remapping version
# video_csv_file = "video_transcripts_with_hierarchy_fixed_1702443584.csv"
# concepts_csv_file = "concept_library_with_stats_1702443584.csv"
# vid_csv_out = "video_transcripts_with_hierarchy_conceptual_lda_1702443584.csv"

# BERT similarity-remapped version
# video_csv_file = "video_transcripts_with_hierarchy_mapped_1702443584.csv"
# concepts_csv_file = "concept_library_mapped_with_stats_1702443584.csv"

# BERT rarity-remapped version
# video_csv_file = "video_transcripts_with_hierarchy_mapped_bert_rarity_1702443584.csv"
# concepts_csv_file = "concept_library_mapped_bert_rarity_with_stats_1702443584.csv"

# <UNK> truncated version
video_csv_file = "video_transcripts_with_hierarchy_mapped_truncated_with_freqs_1702443584.csv"
concepts_csv_file = "concept_library_mapped_truncated_with_stats_1702443584.csv"
vid_csv_out = "video_transcripts_with_hierarchy_mapped_truncated_conceptual_lda_1702443584.csv"

num_topics = 10

# Ensure you have the NLTK data downloaded
nltk.download('stopwords')
nltk.download('wordnet')

def perform_lda(dataframe, num_topics=10, text_col="transcript"):
    # Preprocessing the transcript data
    doc_clean = [lda_preprocess(doc).split() for doc in dataframe[text_col]]  

    # Creating the term dictionary of our corpus, where every unique term is assigned an index
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Creating the LDA model
    lda_model = models.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=50)

    return lda_model, dictionary


concept_df = pd.read_csv(concepts_csv_file)
vid_df = pd.read_csv(video_csv_file)

for index, row in vid_df.iterrows():
  if isinstance(row['activity_concept_hierarchy'], str):
    concept_hierarchy = json.loads(row['activity_concept_hierarchy'])

    # Run the count_concepts function
    primary_concepts, supporting_concepts = count_concepts(concept_hierarchy)

    all_primary_concepts = []
    for concept in primary_concepts.keys():
      if not concept == "<UNK>":
        all_primary_concepts.extend([concept]*primary_concepts[concept])
    vid_df.at[index, "primary_concept_summary"] = ", ".join(all_primary_concepts)

    all_supporting_concepts = []
    for concept in supporting_concepts.keys():
      if not concept == "<UNK>":
        all_supporting_concepts.extend([concept]*supporting_concepts[concept])
    vid_df.at[index, "supporting_concept_summary"] = ", ".join(all_supporting_concepts)

    vid_df.at[index, "concept_summary"] = ", ".join(all_primary_concepts + all_supporting_concepts)


# Train LDA model on concept summary strings
lda_model, term_dict = perform_lda(vid_df, num_topics=num_topics, text_col="concept_summary")

# Display the topics
print("Top words associated with each topic:")
topics = lda_model.print_topics(num_topics=10, num_words=5)
for topic in topics:
    print(topic)

#### We've trained our LDA model. Now, let's get the topic distributions for primary and supporting concepts in each video ####

for i in range(num_topics):
  vid_df[f'topic_{i}'] = 0.0

# Updating the DataFrame with topic weightings
for idx, row in vid_df.iterrows():

  for concept_type in ["primary", "supporting"]:

    # Preprocess the transcript
    processed_txt = lda_preprocess(row[f"{concept_type}_concept_summary"])
    bow = term_dict.doc2bow(processed_txt.split())

    # Get the topic distribution
    topic_dist = get_lda_topic_distribution(lda_model, bow)

    # Update the DataFrame with topic weights
    for topic_id, weight in topic_dist.items():
      vid_df.at[idx, f'{concept_type}_concept_topic_{topic_id}'] = weight

# Now df has additional columns for each topic's weighting
print("DataFrame with topic weightings:")
print(vid_df.head())

# Save the csv with topic distributions, the LDA model, and the term dictionary: 
vid_df.to_csv(vid_csv_out, index=False)
lda_model.save('conceptual_lda_model.model')
term_dict.save('conceptual_lda_term_dict.dict')
