import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

import json
import string
from helpers import count_concepts, lda_preprocess
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


# If conceptual is False, do LDA on the raw transcript. 
# If conceptual is True, do LDA on the extracted concepts.
conceptual = True 
num_topics = 10
lda_of_topics = True

video_csv_file = "video_transcripts_with_hierarchy_mapped_truncated_conceptual_lda_1702443584.csv"

vid_df = pd.read_csv(video_csv_file)

if conceptual:
  tpc = "concept_topic_"
  mat = vid_df[
      ["primary_" + tpc + str(t) for t in range(num_topics)] \
      + ["supporting_" + tpc + str(t) for t in range(num_topics)]
    ].to_numpy()
else: 
  tpc = "topic_"
  mat = vid_df[[tpc + str(t) for t in range(num_topics)]].to_numpy()

# Run t-SNE
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(mat)


# Plot t-SNE results with color coding by creator
vid_df.loc[vid_df['dir_name'] == 'Digital Learning Hub - Imperial College London', 'dir_name'] = 'Imperial College London'
vid_df.loc[vid_df['dir_name'] == 'The Bright Side of Mathematics', 'dir_name'] = 'Bright Side of Math'
vid_df["comp-1"] = tsne_results[:, 0]
vid_df["comp-2"] = tsne_results[:, 1]

plt.figure(figsize=(10, 10))
ax = sns.scatterplot(x="comp-1", y="comp-2", hue="dir_name", data=vid_df)
plt.axis('off')
plt.savefig(f"lda_{tpc}tsne_by_creator.png")
plt.close()

plt.figure(figsize=(6, 4))
ax = sns.scatterplot(x="comp-1", y="comp-2", hue="dir_name", data=vid_df)
ax.get_legend().remove()
plt.axis('off')
plt.savefig(f"lda_{tpc}tsne_by_creator_no_legend.png", dpi=600)
plt.close()



# Get some extra features of vid dfs that can help identify them
theme_list = ["eigen", "system", "transformation", "space", "bas", "invert", "determinant"]

# Function to determine the most frequent theme in a string
def most_frequent_theme(text, themes):
    if isinstance(text, str):
      count_dict = {theme: text.lower().count(theme) for theme in themes}
      most_frequent = max(count_dict, key=count_dict.get)
      # If the highest count is 0, return None or a default value
      return most_frequent if count_dict[most_frequent] > 0 else "none"
    else: 
       return "none"

# Applying the function to each row in the DataFrame
vid_df['theme'] = vid_df.apply(lambda row: most_frequent_theme(row['primary_concept_summary'], theme_list), axis=1)


# Plot t-SNE results with color coding by theme
vid_df["comp-1"] = tsne_results[:, 0]
vid_df["comp-2"] = tsne_results[:, 1]

plt.figure(figsize=(10, 10))
ax = sns.scatterplot(x="comp-1", y="comp-2", hue="theme", data=vid_df)
plt.axis('off')
plt.savefig(f"lda_{tpc}tsne_by_theme.png")
plt.close()

plt.figure(figsize=(6, 4))
ax = sns.scatterplot(x="comp-1", y="comp-2", hue="theme", data=vid_df)
ax.get_legend().remove()
plt.axis('off')
plt.savefig(f"lda_{tpc}tsne_by_theme_no_legend.png", dpi=600)
plt.close()

# if lda_of_topics: 
#   # Ensure NLTK data is available
#   nltk.download('stopwords')
#   nltk.download('wordnet')

#   # Load the saved LDA model and term dictionary
#   lda_model = models.LdaModel.load('conceptual_lda_model.model')
#   term_dict = corpora.Dictionary.load('conceptual_lda_term_dict.dict')

#   concept_list = ["eigenvalues", "determinant"]

#   for idx, row in vid_df.iterrows():
#     if row["concept"] in concept_list:
#       concept = row["concept"]
#       # Preprocess the concept text
#       processed_txt = lda_preprocess(concept)
#       bow = term_dict.doc2bow(processed_txt.split())

#       # Get the topic distribution
#       topic_dist = get_lda_topic_distribution(lda_model, bow)

#       # Update the DataFrame with topic weights
#       for topic_id, weight in topic_dist.items():
#         vid_df.at[idx, f'{concept_type}_concept_topic_{topic_id}'] = weight


