
def video_ids_from_playlist(playlist_id):
    """
    playlist_id: str
        The ID of the playlist from whom the video IDs will be extracted.

    Returns: list of str
        A list of the IDs of the videos in this playlist, in viewing order.
    """
    pass
        

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
    pass


def video_transcript_from_id(video_id):
    """
    Downloads and returns the time signature-less transcript of the video associated
    with the given ID.

    video_id: str
        The ID of the video whose transcript will be downloaded.

    Returns: str
        The transcribed text from the video.
    """
    pass


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