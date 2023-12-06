from playlist_data_scraper import video_ids_from_playlist, video_metadata_from_id
from playlist_data_scraper import video_transcript_from_id, store_playlist_videos_metadata
from googleapiclient.discovery import build

# Creates the YouTube API client
API_KEY = ""
YOUTUBE_CLIENT = build('youtube', 'v3', developerKey=API_KEY)

# Tests the video_ids_from_playlist function
video_ids = video_ids_from_playlist("PLF8mMTQil4Z8vsWfZsb0bcD51Z6JoHPzG")
expected = ("KswZ1wnCejs", "EEQt9ppV4i8", "2XkA8ozbkho", "kQrBbe65YS0")
actual = tuple(video_ids)
assert actual == expected, f"Expected {expected}, but got {actual}."



# Tests the video_metadata_from_id function
#
# When a new key is added to the output as per the
# function specification, add it to this test
metadata = video_metadata_from_id("KswZ1wnCejs", YOUTUBE_CLIENT)
expected = "Intuition for Limits"
actual = metadata["title"]
assert actual == expected, f"Expected metadata[\"title\"] to be {expected}, but got {actual}."

expected = "KswZ1wnCejs"
actual = metadata["id"]
assert actual == expected, f"Expected metadata[\"id\"] to be {expected}, but got {actual}."

expected = "239"
actual = metadata["duration"]
assert actual == expected, f"Expected metadata[\"duration\"] to be {expected}, but got {actual}."

expected = "99"
actual = metadata["views"]
assert int(actual) >= int(expected), f"Expected metadata[\"views\"] to be at least {expected}, but got {actual}."

expected = "7"
actual = metadata["likes"]
assert int(actual) >= int(expected), f"Expected metadata[\"likes\"] to be at least {expected}, but got {actual}."

expected = "Raulz Math World"
actual = metadata["channel_name"]
assert actual == expected, f"Expected metadata[\"channel_name\"] to be {expected}, but got {actual}."

expected = "UC2YE5AUXk8KCGPLOfY9eCXA"
actual = metadata["channel_id"]
assert actual == expected, f"Expected metadata[\"channel_id\"] to be {expected}, but got {actual}."



# Tests the video_transcript_from_id function
transcript = video_transcript_from_id("KswZ1wnCejs")
assert transcript[:3] == "Hey", "Expected first word in transcript to be \"Hey\""
assert transcript[-5:] == "want.", "Expected last word in transcript to be \"want.\""



# DO MANUAL TEST OF video_comments_from_id FUNCTION WHEN TESTING store_playlist_videos_metadata



# Tests the store_playlist_video_metadata function
store_playlist_videos_metadata("PLF8mMTQil4Z8vsWfZsb0bcD51Z6JoHPzG", YOUTUBE_CLIENT, "data_scraper/")
with open("data_scraper/PLF8mMTQil4Z8vsWfZsb0bcD51Z6JoHPzG 000.txt", "r") as file:
    assert file.readline()[:-1] == "PLF8mMTQil4Z8vsWfZsb0bcD51Z6JoHPzG", "Expected correct playlist ID."
    assert file.readline()[:-1] == "0", "Expected correct video playlist position."
    assert file.readline()[:-1] == "4", "Expected correct playlist length."
    assert file.readline()[:-1] == "Intuition for Limits", "Expected correct video title."
    assert file.readline()[:-1] == "KswZ1wnCejs", "Expected correct video ID."
    assert file.readline()[:-1] == "239", "Expected correct video duration."
    assert int(file.readline()[:-1]) >= 99, "Expected correct view count."
    assert int(file.readline()[:-1]) >= 7, "Expected correct like count."
    assert file.readline()[:-1] == "Raulz Math World", "Expected correct channel name."
    assert file.readline()[:-1] == "UC2YE5AUXk8KCGPLOfY9eCXA", "Expected correct channel ID."
    assert file.readline()[:3] == "Hey", "Expected correct transcript."
# Manual Testing Strategy:
# Check the transcript and comments to see if they are correct/reasonable


print("All tests passed!")