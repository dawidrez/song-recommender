from django.shortcuts import render
from django.views.decorators.http import require_http_methods
import os
from django.conf import settings
from .helpers import process_audio

@require_http_methods(["GET", "POST"])
def upload_audio(request):
    if request.method == 'POST':
        audio_file = request.FILES.get('audio_file')
        if audio_file:
            if audio_file.name.endswith('.mp3') or audio_file.name.endswith('.wav'):
                # Save the file to media/songs directory
                songs_dir = os.path.join(settings.MEDIA_ROOT, 'songs')
                file_path = os.path.join(songs_dir, audio_file.name)
                with open(file_path, 'wb+') as destination:
                    for chunk in audio_file.chunks():
                        destination.write(chunk)
                similar_songs, spectrogram_url = process_audio(file_path)
                # Prepare audio information for the template
                audio_type = 'mp3' if audio_file.name.endswith('.mp3') else 'wav'
                audio_url = f"{settings.MEDIA_URL}songs/{audio_file.name}"
                
                return render(request, 'song_recommender/upload.html', {
                    'success': 'File uploaded successfully!',
                    'audio_url': audio_url,
                    'audio_name': audio_file.name,
                    'audio_type': audio_type,
                    'spectrogram_url': spectrogram_url,
                    'similar_songs': similar_songs
                })
            else:
                return render(request, 'song_recommender/upload.html', {
                    'error': 'Invalid file format. Please upload an MP3 or WAV file.'
                })
    return render(request, 'song_recommender/upload.html')
