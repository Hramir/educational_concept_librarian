from playlist_data_scraper import video_ids_from_playlist, video_metadata_from_id
from playlist_data_scraper import video_transcript_from_id, store_playlist_videos_metadata

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
expected = "99"
actual = metadata["views"]
assert int(actual) >= int(expected), f"Expected metadata[\"views\"] to be at least {expected}, but got {actual}."
expected = "UC2YE5AUXk8KCGPLOfY9eCXA"
actual = metadata["channel"]
assert actual == expected, f"Expected metadata[\"channel\"] to be {expected}, but got {actual}."



# Tests the video_transcript_from_id function
transcript = video_transcript_from_id("KswZ1wnCejs")
for word in transcript:
    for char in " \n\t\r":
        assert char not in word, f"Expected no whitespace in words, but found whitespace in {word}."
assert transcript[0] == "Hey", "Expected first word in transcript to be \"Hey\""
assert transcript[-1] == "want.", "Expected last word in transcript to be \"want.\""



# Tests the store_playlist_video_metadata function
store_playlist_videos_metadata("PLF8mMTQil4Z8vsWfZsb0bcD51Z6JoHPzG", "data_scraper/")
with open("data_scraper/PLF8mMTQil4Z8vsWfZsb0bcD51Z6JoHPzG 0.txt", "r") as file:
    assert file.readline()[:-1] == "KswZ1wnCejs", "Expected correct video ID."
    assert file.readline()[:-1] == "239", "Expected correct video duration."
    assert int(file.readline()[:-1]) >= 99, "Expected correct view count."
    assert file.readline()[:-1] == "UC2YE5AUXk8KCGPLOfY9eCXA", "Expected correct channel ID."
# Manual Testing Strategy:
# Check lines after the metadata tested above to see if they match the transcript
# from the video


print("All tests passed!")