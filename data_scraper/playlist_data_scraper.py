from youtubesearchpython import Playlist, Video
from youtube_transcript_api import YouTubeTranscriptApi

def video_ids_from_playlist(playlist_id):
    """
    playlist_id: str
        The ID of the playlist from whom the video IDs will be extracted.

    Returns: list of str
        A list of the IDs of all videos in this playlist, in viewing order.
    """
    
    # Loads the playlist videos
    playlist = Playlist(f"https://www.youtube.com/playlist?list={playlist_id}")

    # Stores all videos' IDs
    video_ids = []
    while True:
        video_ids += [video["id"] for video in playlist.videos]

        # Exits when there are no more videos left
        if not playlist.hasMoreVideos:
            break
            
        # Gets next videos
        playlist.getNextVideos()

    return video_ids
        

def video_metadata_from_id(video_id, youtube_client):
    """
    Downloads and returns difference pieces of metadata about the video associated
    with the given ID.

    video_id: str
        The ID of the video whose metadata will be extracted.
    youtube_client: Resource
        The YouTube API Client built using an API key.

    Returns: dict mapping str to str
        A dictionary that maps a metadata label to the corresponding metadata.

        For backwards compatibility, never remove old dict keys from the implementation;
        instead, keep all old keys and only ever add new keys.

        Currently, the keys in the dict are:
            title: The title of the video
            id: The ID of the video
            duration: The number of seconds the video lasts
            views: The number of views the video has
            likes: The number of likes the video has
            channel_name: The name of the channel that posted the video
            channel_id: The ID of the channel that posted the video
    """

    # Loads the video metadata (excluding like count)
    video_info = Video.getInfo(video_id)

    # Prepares a request for extracting video likes
    request = youtube_client.videos().list(
        part='statistics',
        id=video_id
    )
    response = request.execute()

    return {
        "title": video_info["title"],
        "id": video_info["id"],
        "duration": video_info["duration"]["secondsText"],
        "views": video_info["viewCount"]["text"],
        "likes": response['items'][0]['statistics']['likeCount'],
        "channel_name": video_info["channel"]["name"],
        "channel_id": video_info["channel"]["id"],
    }

def video_transcript_from_id(video_id):
    """
    Downloads and returns the time signature-less transcript of the video associated
    with the given ID.

    video_id: str
        The ID of the video whose transcript will be downloaded.

    Returns: str
        The words of the video's transcribed text, in the order they were transcribed.
        May include substrings that are not English words (e.g., mathematical expressions).
        Words are split and "glued" together with a space character wherever a whitespace
        character appears in the original YouTube transcript. Punctuation used is still kept
        in the word string.

        If the video lacks a transcript, the string "NO TRANSCRIPT AVAILABLE" is returned instead.
    """
    
    # Downloads the transcript data
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=("en", "en-US"))
    except:
        print("No transcript found for", video_id)
        return "NO TRANSCRIPT AVAILABLE"

    # Stores each individual word in the transcript
    dirty_words = []
    for line in transcript:
        dirty_words += line["text"].split()

    # Removes non-ASCII characters from the transcript
    words = []
    for word in dirty_words:
        new_word = ""
        for char in word:
            if char.isascii():
                new_word += char
        words.append(new_word)

    # Glues words together with a space character between them
    complete_transcript = words[0]
    for word in words[1:]:
        if len(word) > 0:
            complete_transcript += " " + word
    return complete_transcript

def video_comments_from_id(video_id, youtube_client):
    """
    Downloads and returns the top-level comments on the video with the given video ID.

    video_id: str
        The ID of the video whose comments will be downloaded.
    youtube_client: Resource
        The YouTube API Client built using an API key.

    Returns: list of str
        The comments on the video with the given video ID, where all newline characters have
        been replaced with space characters.
    """

    # Stores values for the function
    comments = []
    nextPageToken = None

    # Goes through each page of comments
    while True:

        # Requests the data of the video
        response = youtube_client.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            pageToken=nextPageToken
        ).execute()

        # Runs through each comment
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']

            # Replaces newlines in the comments with spaces
            mod_comment = ""
            for comment_char in comment.strip():
                if comment_char.isascii():
                    if comment_char == "\n":
                        mod_comment += " "
                    else:
                        mod_comment += comment_char

            comments.append(mod_comment)

        # Goes to the next page of comments, if any
        nextPageToken = response.get('nextPageToken')
        if not nextPageToken:
            break

    return comments


def store_playlist_videos_metadata(playlist_id, youtube_client, local_path_str=""):
    """
    Stores the metadata (including transcript) from each video in the given playlist
    at the path provided in a text file named:
        "[PLAYLIST_ID] [INDEX OF VIDEO IN PLAYLIST].txt".

    Below is the format of each video metadata file:

    Line 1: Playlist ID
    Line 2: Position (0-indexed) of the video in the playlist
    Line 3: Total number of videos in the playlist from which the video came
    Line 4: Video title
    Line 5: Video ID
    Line 6: Video duration in seconds
    Line 7: Number of views the video has, as of downloading the data
    Line 8: Number of likes the video has, as of downloading the data
    Line 9: Channel name
    Line 10: Channel ID
    Line 11: Transcript

    Lines 12+: Top-level comments on the video, where each line corresponds to an
    individual top-level comment on the video. The replies to comments are not included.

    playlist_id: str
        The ID of the playlist from whom the videos' metadata will be stored.
    youtube_client: Resource
        The YouTube API Client built using an API key.
    local_path_str: str
        The local path to the directory at which the files will be stored
        (without "/" at the end), or "" if the root directory.
    """

    # Checks correctness of path
    if (len(local_path_str) > 0 and local_path_str[-1] == "/"):
        local_path_str = local_path_str[:-1]

    # Runs through each video in the playlist
    video_ids = video_ids_from_playlist(playlist_id)
    for index, video_id in enumerate(video_ids):

        # Stores the info from the video
        metadata = video_metadata_from_id(video_id, youtube_client)
        transcript = video_transcript_from_id(video_id)
        comments = video_comments_from_id(video_id, youtube_client)

        # Writes the video data as specified
        with open(f"{local_path_str}{('/' if local_path_str!='' else '')}{playlist_id} {str(index).zfill(3)}.txt", "w") as file:

            # Stores the playlist ID
            file.write(playlist_id + "\n")

            # Stores the video's playlist position/length
            file.write(str(index) + "\n")
            file.write(str(len(video_ids)) + "\n")

            # Stores the non-transcript/comment metadata
            for tag in ["title", "id", "duration", "views", "likes", "channel_name", "channel_id"]:
                file.write(metadata[tag] + "\n")

            # Stores the video's transcribed words
            file.write(transcript + "\n")

            # Stores the video's top-level comments
            for comment in comments:
                file.write(comment + "\n")
        