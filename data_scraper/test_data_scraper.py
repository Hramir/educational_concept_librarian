from playlist_data_scraper import video_ids_from_playlist, video_metadata_from_id

# Tests the video_ids_from_playlist function
video_ids = video_ids_from_playlist("PLF8mMTQil4Z8vsWfZsb0bcD51Z6JoHPzG")
expected = ("KswZ1wnCejs", "EEQt9ppV4i8", "2XkA8ozbkho", "kQrBbe65YS0")
actual = tuple(video_ids)
assert actual == expected, f"Expected {expected}, but got {actual}."
