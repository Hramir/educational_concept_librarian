import os
import pandas as pd

def parse_transcript(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    video_id = lines[0].strip()
    duration_sec = int(lines[1].strip())
    view_count = int(lines[2].strip())
    channel_id = lines[3].strip()
    transcript = ' '.join(line.strip() for line in lines[4:])

    return video_id, duration_sec, view_count, channel_id, transcript

def process_directory(directory_path):
    data = []

    for subdir, _, files in os.walk(directory_path):
        dir_name = os.path.basename(subdir).replace('data_', '')

        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(subdir, file)
                video_id, duration_sec, view_count, channel_id, transcript = parse_transcript(file_path)
                data.append({
                    'video_id': video_id, 
                    'duration_sec': duration_sec,
                    'view_count': view_count,
                    'channel_id': channel_id,
                    'dir_name': dir_name, 
                    'transcript': transcript
                })

    return pd.DataFrame(data)