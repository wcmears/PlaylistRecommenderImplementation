import spotipy
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import re
import itertools
import random


def get_recommended_songs(playlist_link):

    client_id = "632027a5f07242acbc2b319d1876fbac"
    client_secret = "c66533e50f0e46cf910e46a93f58c811"
    username = "will.mears50"

    #note that I extended the scope to also modify non-public playlists
    scope = "playlist-modify-public playlist-modify-private playlist-read-private playlist-read-collaborative"

    redirect_uri = "http://localhost/"

    client_cred_manager = SpotifyClientCredentials(client_id=client_id, 
                                                        client_secret=client_secret)
                                                        
    sp = spotipy.Spotify(client_credentials_manager = client_cred_manager)

    token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
    if token:
        sp = spotipy.Spotify(auth=token)
    
    # Get the playlist ID from the playlist link
    playlist_id = re.search(r"playlist\/([\w\d]+)", playlist_link).group(1)
    
    # Get the list of track IDs from the playlist
    playlist_tracks = sp.playlist_tracks(playlist_id)
    track_ids = [track['track']['id'] for track in playlist_tracks['items']]

    # Choose random combinations of 5 tracks as seeds
    num_combinations = 5  # Change this value to control the number of requests
    seed_combinations = random.sample(list(itertools.combinations(track_ids, 5)), num_combinations)

    # Get the list of recommended track IDs from multiple requests
    recommended_track_ids = set()
    for seed_tracks in seed_combinations:
        recommended_tracks = sp.recommendations(seed_tracks=seed_tracks, limit=20)  # Lower the limit to distribute results
        for track in recommended_tracks['tracks']:
            recommended_track_ids.add(track['id'])

    # Convert the set to a list
    recommended_track_ids = list(recommended_track_ids)
    
    playlist_details = sp.playlist(playlist_id)

    # Create a new playlist with the recommended tracks
    playlist_name = f"Recommended songs for {playlist_details['name']}"
    playlist = sp.user_playlist_create(user=sp.me()['id'], name=playlist_name, public=False)
    sp.user_playlist_add_tracks(user=sp.me()['id'], playlist_id=playlist['id'], tracks=recommended_track_ids)
    
    # Get the link to the new playlist
    playlist_link = playlist['external_urls']['spotify']
    
    return playlist_link
