# IMPORTANT: run this file from the project root directly, like "python utils/make_df.py"

import os
import pandas as pd

def process_directory(directory_path):
  data = []
  for file in os.listdir(directory_path):
    if file.endswith('.txt'):
      file_path = os.path.join(directory_path, file)
      with open(file_path, 'r') as file:
        lines = file.readlines()
      data.append({
        'video_id': lines[4].strip(),
        'video_title': lines[3].strip(), 
        'duration_sec': int(lines[5].strip()), 
        'view_count': int(lines[6].strip()),
        'like_count': int(lines[7].strip()),
        'like_to_view_ratio': float(int(lines[7].strip())) / int(lines[6].strip()),
        'playlist_id': lines[0].strip(),
        'channel_id': lines[9].strip(),
        'playlist_index': int(lines[1].strip()),
        'playlist_length': int(lines[2].strip()),
        'playlist_position': (int(lines[1].strip()) + 0.5) / int(lines[2].strip()),
        'dir_name': lines[8].strip(), 
        'transcript': lines[10].strip(),
        'comments': [comment.replace("\n", "") for comment in lines[11:]],
      })

  return pd.DataFrame(data)

# Load the data
directory_path = 'data/dataset_v2'
df = process_directory(directory_path)
print("Dataframe head:")
print(df.head())
print("----------------------------------")

print("Number of duplicate videos:", df.duplicated(subset=['video_id', 'playlist_id']).sum())

df = df.drop_duplicates(subset=['video_id', 'playlist_id'])

df_sorted = df.sort_values(by=['playlist_id', 'playlist_index'])

df_sorted.to_csv("video_transcripts.csv", index=False)
