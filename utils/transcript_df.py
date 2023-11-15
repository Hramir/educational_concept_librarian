import os
import pandas as pd

def parse_transcript(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    video_id = lines[0].strip()
    transcript = ' '.join(line.strip() for line in lines[4:])

    return video_id, transcript

def process_directory(directory_path):
    data = []

    for subdir, _, files in os.walk(directory_path):
        playlist_name = os.path.basename(subdir).replace('data_', '')

        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(subdir, file)
                video_id, transcript = parse_transcript(file_path)
                data.append({'video_id': video_id, 'transcript': transcript, 'playlist_name': playlist_name})

    return pd.DataFrame(data)