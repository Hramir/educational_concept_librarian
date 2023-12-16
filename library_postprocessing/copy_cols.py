import pandas as pd

## VERSION FOR COPYING LDA TOPICS
# # Define the path to the source dataframe and other dataframes
# source_file = 'video_transcripts_with_hierarchy_mapped_truncated_1702443584.csv'
# dest_files = [
#   "video_transcripts_with_hierarchy_1702443584.csv",
#   "video_transcripts_with_hierarchy_fixed_1702443584.csv", 
#   "video_transcripts_with_hierarchy_mapped_1702443584.csv", 
#   "video_transcripts_with_hierarchy_mapped_bert_rarity_1702443584.csv",
# ]
# # Define the columns to be copied
# num_topics = 10
# cols = ["topic_" + str(t) for t in range(num_topics)]

## VERSION FOR COPYING SENTIMENTS
# Define the path to the source dataframe and other dataframes
source_file = "video_transcripts_comment_sentiment.csv"
dest_files = [
  "video_transcripts_with_hierarchy_1702443584.csv",
  "video_transcripts_with_hierarchy_fixed_1702443584.csv", 
  "video_transcripts_with_hierarchy_mapped_1702443584.csv", 
  "video_transcripts_with_hierarchy_mapped_bert_rarity_1702443584.csv",
  "video_transcripts_with_hierarchy_mapped_truncated_1702443584.csv",
]
cols = ["sentiments", "average_sentiment"]

# Load the source dataframe
source_df = pd.read_csv(source_file)

# Iterate over each target file
for file in dest_files:
  # Load the target dataframe
  target_df = pd.read_csv(file)

  # Check if 'video_id' column exists in both dataframes
  if 'video_id' in source_df.columns and 'video_id' in target_df.columns:
    # Merge the dataframes on the 'video_id' column, including only the columns from the source
    merged_df = pd.merge(target_df, source_df[cols + ['video_id']], on='video_id', how='left')

    # Save the merged dataframe back to CSV
    merged_df.to_csv(file, index=False)
  else:
    print(f"'video_id' column not found in both source and target DataFrame for file: {file}")
