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

        If the video lacks a transcript, a one-element list ["NO TRANSCRIPT AVAILABLE"]
        is returned instead.
    """
    
    # Downloads the transcript data
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=("en", "en-US"))
    except:
        return ["NO TRANSCRIPT AVAILABLE"]

    # Stores each individual word in the transcript
    words = []
    for line in transcript:
        words += line["text"].split()

    return words


def store_playlist_videos_metadata(playlist_id, local_path_str=""):
    """
    Stores the metadata (including transcript) from each video in the given playlist
    at the path provided in a text file named:
        "[PLAYLIST_ID] [INDEX OF VIDEO IN PLAYLIST].txt".

    Below is the format of each video metadata file:

    Line 1: Video ID
    Line 2: Video duration in seconds
    Line 3: The number of views the video has as of downloading the data
    Line 4: Channel ID

    Lines 5+: The words (split as in video_transcript_from_id) from the video's
    transcript, each on its own line.

    The number of lines from the beginning of the file to the point at which the
    transcribed words begin is equal to the number of keys in the dictionary returned by
    the video_metadata_from_id function.

    playlist_id: str
        The ID of the playlist from whom the videos' metadata will be stored.
    local_path_str: str
        The local path to the directory at which the files will be stored
        (without "/" at the end), or "" if the root directory.
    """

    # Checks correctness of path
    if (len(local_path_str) > 0 and local_path_str[-1] == "/"):
        local_path_str = local_path_str[:-1]

    # Runs through each video in the playlist
    for index, video_id in enumerate(video_ids_from_playlist(playlist_id)):

        # Stores the info from the video
        metadata = video_metadata_from_id(video_id)
        transcript = video_transcript_from_id(video_id)

        # Writes the video data as specified
        with open(f"{local_path_str}{('/' if local_path_str!='' else '')}{playlist_id} {index}.txt", "w") as file:

            # Stores the non-transcript metadata
            for tag in ["id", "duration", "views", "channel"]:
                file.write(metadata[tag] + "\n")

            # Stores the video's transcribed words
            for word in transcript:
                file.write(word + "\n")
        