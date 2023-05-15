import os
import librosa
import numpy as np
import json
import statistics as s
from aubio import pitch


class GetSoundStats:

    def __init__(self):
        self.mfcc = None
        self.chroma = None

    def getStats(self, y, sr):
        # Calculate the spectral centroid
        spectralCentroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        # Calculate the spectral rolloff
        spectralRolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Calculate the short-time Fourier transform (STFT)
        hop_length = 512
        n_fft = 2048
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        # Calculate the magnitude spectrogram
        mag_spec = librosa.magphase(stft)[0]

        # Calculate the spectral flux
        spectral_flux = librosa.feature.delta(mag_spec, order=1)

        # Calculate the ZCR
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

        # Calculate the spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)


        pitch_method = 'yin'
        frame_size = 512
        hop_size = 256  # For 50% overlap
        tolerance = 0.8


        # Calculate the onset strength
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)

        # Calculate the energy
        energy = sum(abs(y)**2)

        # Calculate the entropy of energy
        frame_size_ms = 30  # frame size in milliseconds
        frame_size = int(frame_size_ms * sr / 1000)  # frame size in samples
        energy = sum(abs(y)**2)
        entropy = 0.0
        for i in range(0, len(y), frame_size):
            frame_energy = sum(abs(y[i:i+frame_size])**2) / (energy + 1e-6)  # add a small number to prevent division by zero
            p = frame_energy
            if p > 0:  # only calculate entropy for non-zero probabilities
                entropy -= p * np.log2(p)
        
        # Calculate the MFCCs with 12 coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        self.mfcc = np.array(mfcc)

        # Calculate the chroma vector with 12 pitch classes
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)

        self.chroma = np.array(chroma)

        # Create a dictionary to hold all of the sound stats
        mfcc = self.mfcc.tolist()
        chroma = self.chroma.tolist()
        soundStats = {
            'spectralCentroidMean': s.mean(spectralCentroid.tolist()[0]),
            'spectralCentroidStd': s.stdev(spectralCentroid.tolist()[0]),
            'spectralRolloffMean': s.mean(spectralRolloff.tolist()[0]),
            'spectralRolloffStd': s.stdev(spectralRolloff.tolist()[0]),            
            'spectralFluxMean': s.mean(spectral_flux.tolist()[0]),
            'spectralFluxSTD': s.stdev(spectral_flux.tolist()[0]),
            'zeroCrossingRateMean': s.mean(zero_crossing_rate.tolist()[0]),
            'zeroCrossingRateStd': s.stdev(zero_crossing_rate.tolist()[0]),
            'energy': energy,
            'entropy': entropy,

            'mfcc1Mean': s.mean(mfcc[0]),
            'mfcc2Mean': s.mean(mfcc[1]),
            'mfcc3Mean': s.mean(mfcc[2]),
            'mfcc4Mean': s.mean(mfcc[3]),
            'mfcc5Mean': s.mean(mfcc[4]),
            'mfcc6Mean': s.mean(mfcc[5]),
            'mfcc7Mean': s.mean(mfcc[6]),
            'mfcc8Mean': s.mean(mfcc[7]),
            'mfcc9Mean': s.mean(mfcc[8]),
            'mfcc10Mean': s.mean(mfcc[9]),
            'mfcc11Mean': s.mean(mfcc[10]),
            'mfcc12Mean': s.mean(mfcc[11]),
            'mfcc13Mean': s.mean(mfcc[12]),

            'mfcc1stdev': s.stdev(mfcc[0]),
            'mfcc2stdev': s.stdev(mfcc[1]),
            'mfcc3stdev': s.stdev(mfcc[2]),
            'mfcc4stdev': s.stdev(mfcc[3]),
            'mfcc5stdev': s.stdev(mfcc[4]),
            'mfcc6stdev': s.stdev(mfcc[5]),
            'mfcc7stdev': s.stdev(mfcc[6]),
            'mfcc8stdev': s.stdev(mfcc[7]),
            'mfcc9stdev': s.stdev(mfcc[8]),
            'mfcc10stdev': s.stdev(mfcc[9]),
            'mfcc11stdev': s.stdev(mfcc[10]),
            'mfcc12stdev': s.stdev(mfcc[11]),
            'mfcc13stdev': s.stdev(mfcc[12]),

            'chroma1Mean': s.mean(chroma[0]),
            'chroma2Mean': s.mean(chroma[1]),
            'chroma3Mean': s.mean(chroma[2]),
            'chroma4Mean': s.mean(chroma[3]),
            'chroma5Mean': s.mean(chroma[4]),
            'chroma6Mean': s.mean(chroma[5]),
            'chroma7Mean': s.mean(chroma[6]),
            'chroma8Mean': s.mean(chroma[7]),
            'chroma9Mean': s.mean(chroma[8]),
            'chroma10Mean': s.mean(chroma[9]),
            'chroma11Mean': s.mean(chroma[10]),
            'chroma12Mean': s.mean(chroma[11]),

            'chroma1stdev': s.stdev(chroma[0]),
            'chroma2stdev': s.stdev(chroma[1]),
            'chroma3stdev': s.stdev(chroma[2]),
            'chroma4stdev': s.stdev(chroma[3]),
            'chroma5stdev': s.stdev(chroma[4]),
            'chroma6stdev': s.stdev(chroma[5]),
            'chroma7stdev': s.stdev(chroma[6]),
            'chroma8stdev': s.stdev(chroma[7]),
            'chroma9stdev': s.stdev(chroma[8]),
            'chroma10stdev': s.stdev(chroma[9]),
            'chroma11stdev': s.stdev(chroma[10]),
            'chroma12stdev': s.stdev(chroma[11]),

            'spectralContrastMean': np.mean(spectral_contrast),
            'spectralContrastStd': np.std(spectral_contrast),
            'onsetStrengthMean': np.mean(onset_strength),
            'onsetStrengthStd': np.std(onset_strength),
    
        }

        return soundStats

def process_song_reccomend(playlist_dir, song):
    audio_extensions = ['.mp3', '.wav', '.flac', '.m4a']
    if song.endswith(tuple(audio_extensions)):
        song_path = os.path.join(playlist_dir, song)

        # Load the audio file using librosa
        y, sr = librosa.load(song_path)

        # Get sound stats using GetSoundStats class
        stats = GetSoundStats().getStats(y, sr)

        return stats