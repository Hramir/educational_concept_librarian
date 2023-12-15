import pandas as pd
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine
from tqdm import tqdm
import pickle
from helpers import get_embedding

model_id = 'tbs17/MathBERT'
concept_library_pkl = "concept_library_1702443584.pkl"
embedding_dict_pkl = "BERT_embeddings_for_concepts_1702443584.pkl"

with open(concept_library_pkl, 'rb') as file:
  concept_library = pickle.load(file)

model = BertModel.from_pretrained(model_id)
tokenizer = BertTokenizer.from_pretrained(model_id)

print(f"Getting embeddings for all {len(concept_library)} concepts")
embedding_dict = {}
for concept in tqdm(concept_library):
  embedding_dict[concept] = get_embedding(model, tokenizer, concept).detach().numpy()

print("Shape of embeddings:", embedding_dict[list(concept_library)[0]].shape)

with open(embedding_dict_pkl, "wb") as file:
  pickle.dump(embedding_dict, file)