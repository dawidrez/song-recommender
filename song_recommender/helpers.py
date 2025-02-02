from django.conf import settings
import librosa
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import faiss
from typing import NamedTuple
import tensorflow as tf

class TrackInfo(NamedTuple):
   path: str
   title: str
   spectrogram_url: str
   audio_url: str
   audio_type: str

def create_spectrogram(path: str) -> np.array:
   """Use librosa to extract spectrogram from audio file"""
   y, sr = librosa.load(path, sr=44100)
   spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
   spec_dB = librosa.power_to_db(spec, ref=np.max)
   spec_clipped = np.clip(spec_dB, -80, 0)
   spec_resized = cv2.resize(spec_clipped, (256, 256), interpolation=cv2.INTER_AREA)
   spec_normalized = (spec_resized + 80) / 80

   return spec_normalized


def visualize_spectrogram(spectrogram: np.array, title="Spectrogram") -> str:
   plt.figure(figsize=(10, 6))
   plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
   plt.colorbar(format='%+2.0f dB')
   plt.title(title)
   plt.xlabel('Time')
   plt.ylabel('Mel Frequency')
   spectrograms_dir = os.path.join(settings.MEDIA_ROOT, 'spectrograms')
   updated_title = title.split(".")[:-1]
   updated_title = "".join(updated_title)
   spectrograms_url = f"{spectrograms_dir}/{updated_title}.png"
   plt.savefig(spectrograms_url)
   plt.close()
   return f"{settings.MEDIA_URL}/spectrograms/{updated_title}.png"


def process_audio(path: str)-> tuple[list[TrackInfo],str]:
   spec = create_spectrogram(path)
   filename = os.path.basename(path)
   spectrogram_url = visualize_spectrogram(spec, filename)
   similar_songs = get_similar_songs(spec)
   return similar_songs, spectrogram_url


def load_index():
    index = faiss.read_index("song_recommender/AI/index.bin")
    return index

def get_encoder():
    encoder = tf.keras.models.load_model("song_recommender/AI/encoder_model8.h5")
    return encoder

def get_embedding(spectrogram):
   combined_array = np.stack([spectrogram], axis=0)
   encoder = get_encoder()
   embedding = encoder.predict(combined_array)
   return embedding

def get_similar_songs(spectrogram):
    # Get embedding for the input spectrogram
    embedding = get_embedding(spectrogram)
    index = load_index()
    _, indices = index.search(embedding.reshape(1, -1), 5)
    indices = indices[0]
    tracks = os.listdir(os.path.join(settings.BASE_DIR, 'static/music'))
    sorted_tracks = sorted(tracks)
    similar_songs = []
    for index in indices:
        track_name = sorted_tracks[index]
        base_name = os.path.splitext(track_name)[0]
        similar_songs.append(TrackInfo(
            path=os.path.join(settings.BASE_DIR, 'static/music', track_name),
            title=base_name,
            spectrogram_url=f"{settings.STATIC_URL}images/{base_name}.png",
            audio_url=f"{settings.STATIC_URL}music/{track_name}",
            audio_type='mp3' if track_name.endswith('.mp3') else 'wav'
        ))
    print(similar_songs)
    return similar_songs


def create_index():
    embedding_array = np.load("AI/embeddings_spectrograms.npy")
    index = faiss.IndexFlatL2(embedding_array.shape[1])
    index.add(embedding_array)
    faiss.write_index(index, "AI/index.bin")

def read_spectrogram(filepath: str) -> np.array:
    spectrogram = np.load(filepath)

    spectrogram = np.expand_dims(spectrogram, axis=-1)
    return spectrogram

def read_spectrogram_image(filepath: str) -> np.array:
    spectrogram = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    spectrogram = cv2.resize(spectrogram, (256, 256), interpolation=cv2.INTER_AREA)
    spectrogram = spectrogram / 255.0
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    print(spectrogram.shape)
    return spectrogram


#encoder = tf.keras.models.load_model('AI/encoder_model8.h5')
#spectrograms = []
#tracks = os.listdir("../static/spectrograms")
#sorted_tracks = sorted(tracks)
#for track in sorted_tracks:
#    file_path = os.path.join("../static/spectrograms", track)
#    spectrograms.append(read_spectrogram(file_path))
#
#combined_array = np.stack(spectrograms, axis=0)
#embedding_spectograms = encoder.predict(combined_array)
#
#np.save("AI/embeddings_spectrograms.npy", embedding_spectograms)

#create_index()