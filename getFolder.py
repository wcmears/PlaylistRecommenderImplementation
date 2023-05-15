import subprocess
import os
import re

def getPlaylistFolder(playlistLink):
    # extract the playlist ID from the link
    if 'spotify:playlist:' in playlistLink:
        playlistID = playlistLink.split(":")[-1]
    elif 'open.spotify.com' in playlistLink:
        playlistID = re.search(r'playlist/(\w+)', playlistLink).group(1)
    else:
        raise ValueError("Invalid playlist link format")

    # create a folder with the playlist ID
    folderPath = f"{playlistID}_mp3"
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    # use subprocess to run the spotdl command and save the songs in the folder
    subprocess.run(f"spotdl --playlist {playlistLink} --output ./{folderPath}/ --format mp3", shell=True)

    return folderPath


