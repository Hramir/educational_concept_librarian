
from youtubesearchpython import Playlist, Video
from youtube_transcript_api import YouTubeTranscriptApi

def video_ids_from_playlist(playlist_id):
    """
    playlist_id: str
        The ID of the playlist from whom the video IDs will be extracted.

    Returns: list of str
        A list of the IDs of the videos in this playlist, in viewing order.
    """
    
    # Loads the playlist videos
    videos = Playlist(f"https://www.youtube.com/playlist?list={playlist_id}").videos

    return [video["id"] for video in videos]
        

def video_metadata_from_id(video_id):
    """
    Downloads and returns difference pieces of metadata about the video associated
    with the given ID.

    video_id: str
        The ID of the video whose metadata will be extracted.

    Returns: dict mapping str to str
        A dictionary that maps a metadata label to the corresponding metadata.

        For backwards compatibility, never remove old dict keys from the implementation;
        instead, keep all old keys and only ever add new keys.

        Currently, the keys in the dict are:
            id: The ID of the video
            duration: The number of seconds the video lasts
            views: The number of views the video has
            channel: The ID of the channel that posted the video
    """

    # Loads the video metadata
    video_info = Video.getInfo(video_id)

    return {
        "id": video_info["id"],
        "duration": video_info["duration"]["secondsText"],
        "views": video_info["viewCount"]["text"],
        "channel": video_info["channel"]["id"],
    }


def video_transcript_from_id(video_id):
    """
    Downloads and returns the time signature-less transcript of the video associated
    with the given ID.

    video_id: str
        The ID of the video whose transcript will be downloaded.

    Returns: list of str
        The words of the video's transcribed text, in the order they were transcribed.
        May include strings that are not English words (e.g., mathematical expressions).
        Words are split wherever a whitespace character appears in the YouTube transcript.
        Punctuation used is still kept in the word string.
    """
    
    # Downloads the transcript data
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=("en", "en-US"))

    # Stores each individual word in the transcript
    words = []
    for line in transcript:
        words += line["text"].split()

    return words


def store_playlist_videos_metadata(playlist_id, path_str):
    """
    Stores the metadata (including transcript) from each video in the given playlist
    at the path provided.

    Below is the format of each video metadata file:

    TODO: Fill out structure of video file

    playlist_id: str
        The ID of the playlist from whom the videos' metadata will be stored.
    path_str: str
        The path string to the path at which the metadata files will be saved.
    """
    pass