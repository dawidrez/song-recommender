import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2
from h5py.h5i import DATASET

INPUT_DIRECTORY = "dataset"
OUTPUT_DIRECTORY = "spectograms"

def visualize_spectrogram(spectrogram: np.array, title="Spectrogram"):
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.savefig(f"{OUTPUT_DIRECTORY}/{title}.png")

def create_spectogram(path:str)->np.array:
    y, sr = librosa.load(path, sr=44100)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spec_dB = librosa.power_to_db(spec, ref=np.max)
    spec_clipped = np.clip(spec_dB, -80, 0)
    spec_resized = cv2.resize(spec_clipped, (256, 256), interpolation=cv2.INTER_AREA)
    spec_normalized = (spec_resized + 80) / 80

    return spec_normalized
def save_spec(track_name:str, spec: np.array) -> None:
    np.save(f"{OUTPUT_DIRECTORY}/{track_name}", spec)




tracks = os.listdir(DATASET)
for track in tracks:
  file_path = f"{DATASET}/{track}"
  spec = create_spectogram(file_path)
  visualize_spectrogram(spec, track)




