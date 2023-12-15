import pandas as pd
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine
from tqdm import tqdm
import pickle
from helpers import get_embedding

concept_library_init_file = "concept_library_with_stats_1702443584.csv"
concept_mappings_file = "concept_similarity_mappings_87_1702443584.pkl"
cosine_sim_threshold = 0.87
model_id = 'tbs17/MathBERT'

df = pd.read_csv(concept_library_init_file)

model = BertModel.from_pretrained(model_id)
tokenizer = BertTokenizer.from_pretrained(model_id)

df = df.sort_values(by='total_count', ascending=True).reset_index(drop=True)

concept_mappings = {}

print("Getting embeddings for all concepts")
embedding_dict = {}
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
  concept = row["concept"]
  concept_embedding = get_embedding(model, tokenizer, concept)
  embedding_dict[concept] = concept_embedding

print("Iterating through the DataFrame to get concept mappings")
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
  current_embedding = embedding_dict[row['concept']]

  most_similar = None
  highest_similarity = 0

  # Compare with concepts in rows below
  for j in range(i + 1, len(df)):
    next_row = df.iloc[j]

    if row['concept'] == next_row['concept']:
      print("Warning: duplicated concepts")
      continue

    next_embedding = embedding_dict[next_row['concept']]

    # Cosine similarity
    similarity = 1 - cosine(current_embedding.detach().numpy(), next_embedding.detach().numpy())
    
    if similarity > highest_similarity:
      highest_similarity = similarity
      most_similar = next_row['concept']

  # Check if the highest similarity is above cosine_sim_threshold and add to the mappings
  if highest_similarity > cosine_sim_threshold:
    concept_mappings[row['concept']] = most_similar

print(concept_mappings)
print(len(concept_mappings), "mappings in total")
with open(concept_mappings_file, 'wb') as file:
    pickle.dump(concept_mappings, file)
