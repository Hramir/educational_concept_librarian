import json
import pandas as pd

def make_sentence(st):
  if pd.isna(st):
    return None
  data = st.replace("'", "\"")
  data = json.loads(data)
  ret = ""

  def handle_activity(activity):
    nonlocal ret
    ret += activity['activity']
    ret += ' of '
    ret += activity['primary_concept']
    if activity['supporting_concepts']:
      ret += ' supported by '
      ret += ", ".join(activity['supporting_concepts'])
    if activity['activities']:
      ret += ' with following: '
      for i in range(len(activity['activities'])):
        handle_activity(activity['activities'][i])
        if i != len(activity['activities']) - 1:
          ret += ', '

  for activity in data['lesson']:
    handle_activity(activity)
    ret += ". "
  return ret

if __name__ == "__main__":
    df = pd.read_csv('video_transcripts_with_hierarchy_1701550479.csv')
    df["sentence_embedding"] = df["activity_concept_hierarchy"].apply(make_sentence)
    df.to_csv('sentence_embed.csv')