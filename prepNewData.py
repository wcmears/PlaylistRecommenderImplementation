from getStatsNewFiles import process_song_reccomend
import pandas as pd
import torch

def load_and_preprocess_data(folderPath, song, encoder, scaler):
    soundStats = process_song_reccomend(folderPath, song)
    df = pd.DataFrame([soundStats])
    X = scaler.transform(df)
    X = torch.tensor(X, dtype=torch.float32)
    return X
