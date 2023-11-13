from playlist_data_scraper import video_ids_from_playlist, video_metadata_from_id

# Tests the video_ids_from_playlist function
video_ids = video_ids_from_playlist("PLF8mMTQil4Z8vsWfZsb0bcD51Z6JoHPzG")
expected = ("KswZ1wnCejs", "EEQt9ppV4i8", "2XkA8ozbkho", "kQrBbe65YS0")
actual = tuple(video_ids)
assert actual == expected, f"Expected {expected}, but got {actual}."



# Tests the video_metadata_from_id function
#
# When a new key is added to the output as per the
# function specification, add it to this test
metadata = video_metadata_from_id("KswZ1wnCejs")
expected = "KswZ1wnCejs"
actual = metadata["id"]
assert actual == expected, f"Expected metadata[\"id\"] to be {expected}, but got {actual}."
expected = "239"
actual = metadata["duration"]
assert actual == expected, f"Expected metadata[\"duration\"] to be {expected}, but got {actual}."
expected = "96"
actual = metadata["views"]
assert actual == expected, f"Expected metadata[\"views\"] to be {expected}, but got {actual}."
expected = "UC2YE5AUXk8KCGPLOfY9eCXA"
actual = metadata["channel"]
assert actual == expected, f"Expected metadata[\"channel\"] to be {expected}, but got {actual}."


print("All tests passed!")