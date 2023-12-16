from transformers import pipeline
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
tqdm.pandas()

PIPELINE = None

def get_sentiment_score(comments):
    global PIPELINE
    assert PIPELINE is not None
    if len(comments) == 0:
        return []
    scores = []
    for comment in comments:
        result = PIPELINE(comment, max_length=512)[0]        
        if result['label'] == 'positive':
            scores.append(result['score'])
        elif result['label'] == 'negative':
            scores.append(-result['score'])
        elif result['label'] == 'neutral':
            scores.append(0)
        else:
            raise ValueError
    return scores

def analyze_comments_from_file(file_path, pipeline):
    with open(file_path, 'r') as file:
        
        for _ in range(4):
            next(file)
        video_id = next(file).strip()
        for _ in range(6):
            next(file)
        comments = []
        for line in file:
            comments.append(line.strip())
        if comments[0] == 'NO COMMENTS AVAILABLE':
            return video_id, None
        return video_id, get_sentiment_score(pipeline, comments)

def parse_current_directory():
    txt_files = glob.glob('*.txt')
    data = []
    global PIPELINE
    PIPELINE = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

    for file in txt_files:
        video_id, scores = analyze_comments_from_file(file, sentiment_pipeline)
        if scores:
            avg_score = sum(scores)/len(scores)
            data.append([video_id, scores, avg_score])
    
    df = pd.DataFrame(data, columns=['video_id', 'comment_scores', 'average_score'])
    df.to_csv('comment_data.csv', index=False)


def parse_csv(file_name):
    global PIPELINE
    PIPELINE = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    print('Generated pipeline')
    df = pd.read_csv(file_name)
    df['comments'] = df['comments'].apply(ast.literal_eval)
    print('Calculating sentiments:')
    df['sentiments'] = df['comments'].progress_apply(get_sentiment_score)
    print('Generating means:')
    df['average_sentiment'] = df['sentiments'].progress_apply(np.mean)

    df.to_csv('comment_data.csv', index=False)

if __name__ == "__main__":
    parse_csv('video_transcripts.csv')