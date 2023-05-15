from getStatsMulti import process_song
from concurrent.futures import ThreadPoolExecutor
from getFolder import getPlaylistFolder
import subprocess
import os
from sound_classifier import train_sound_classifier, playlistClassifier
import torch
from recSongs import get_recommended_songs
from prepNewData import load_and_preprocess_data
from sound_classifier import load_trained_model

# ask the user for the playlist link
playlistLink = input("Enter the link to the Spotify playlist: ")

# call the getPlaylistFolder() function which creates a folder with all songs from the playlist link
folderPath = getPlaylistFolder(playlistLink)

matchPlaylist = str(folderPath)

subprocess.run(f"mkdir {folderPath}", shell=True)

audio_extensions = ['.mp3', '.wav', '.flac', '.m4a']

#Take sound stats on this folder
for song in os.listdir(folderPath):
    if song.endswith(tuple(audio_extensions)):
        print("Processing song:", song)
        try:
            process_song(folderPath, song)
        except Exception as e:
            print(f"Error processing song '{song}': {e}")

#Train model on previous playlist data along with new data from new playlist
trained_model, input_size, num_classes, encoder, scaler = train_sound_classifier()

# Load the trained model
model_path = "trained_model.pth"
loaded_model = load_trained_model(model_path, input_size, num_classes)

#Load recommended songs for playlist from spotify
songsToTest = get_recommended_songs(playlistLink)

folderPath = getPlaylistFolder(songsToTest)

#Songs are given a probability percentage for each node. If the song is below 50% for every node, it goes into a node called other
#The plan for future work is to have these songs recycled and kept stored and then this reattempts to classify them as more playlists are added
threshold = 0.5

recommendedSongs = []

#Turn every song into an array of data and put into the neural network
for song in os.listdir(folderPath):
    if song.endswith(tuple(audio_extensions)):
        print("Processing song:", song)
        try:
            X = load_and_preprocess_data(folderPath, song, encoder, scaler)
            output = loaded_model(X)
            max_probability, prediction = torch.max(output, dim=1)
            max_probability = max_probability.item()
            if max_probability >= threshold:
                predicted_playlist = encoder.inverse_transform(prediction)
                print(f"Predicted playlist for '{song}': {predicted_playlist[0]}")
                if predicted_playlist[0] == matchPlaylist:
                    recommendedSongs.append(song)
            else:
                print(f"Predicted playlist for '{song}': Other")
        except Exception as e:
            print(f"Error processing song '{song}': {e}")

print(f"You have {len(recommendedSongs)} recommended songs: {recommendedSongs}")