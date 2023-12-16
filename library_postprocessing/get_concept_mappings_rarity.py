import pandas as pd
from tqdm import tqdm
import pickle

concept_library_init_file = "concept_library_mapped_with_stats_1702443584.csv"
concept_mappings_file = "concept_rarity_mappings_1702443584.pkl"
min_video_count = 5

df = pd.read_csv(concept_library_init_file)

df = df.sort_values(by='total_count', ascending=True).reset_index(drop=True)

print("" in df["concept"])

concept_mappings = {}
print("Iterating through the DataFrame to get concept mappings")
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
  if row["video_count"] < min_video_count:
    concept_mappings[row["concept"]] = "<UNK>"

if "" in df["concept"]:
  concept_mappings[""] = "<UNK>"

print(len(concept_mappings), "mappings to \"<UNK>\" in total")
with open(concept_mappings_file, 'wb') as file:
    pickle.dump(concept_mappings, file)

