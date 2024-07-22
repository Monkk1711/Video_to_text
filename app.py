from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import deepspeech
import numpy as np
import wave
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Ensure directories exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('segments'):
    os.makedirs('segments')

# Load DeepSpeech model
model_path = "C:/Users/lmohi/Desktop/Audio To Text/deepspeech-0.9.3-models.pbmm"
scorer_path = "C:/Users/lmohi/Desktop/Audio To Text/deepspeech-0.9.3-models.scorer"
model = deepspeech.Model(model_path)
model.enableExternalScorer(scorer_path)

def convert_mp4_to_mp3(mp4_file, mp3_file):
    try:
        video_clip = VideoFileClip(mp4_file)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(mp3_file)
        audio_clip.close()
        video_clip.close()
    except Exception as e:
        logging.error(f"Error in convert_mp4_to_mp3: {e}")
        raise

def segment_audio(audio_path, output_dir):
    try:
        song = AudioSegment.from_wav(audio_path)
        duration = 2 * 60 * 1000  # 2 minutes in milliseconds
        segments = [song[start:start+duration] for start in range(0, len(song), duration)]
        segment_paths = []
        for i, segment in enumerate(segments):
            segment_path = os.path.join(output_dir, f"segment_{i+1}.wav")
            segment.export(segment_path, format="wav")
            segment_paths.append(segment_path)
        return segment_paths
    except Exception as e:
        logging.error(f"Error in segment_audio: {e}")
        raise

def read_wave(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return sample_rate, audio
    except Exception as e:
        logging.error(f"Error in read_wave: {e}")
        raise

def transcribe(audio_path):
    try:
        sample_rate, audio = read_wave(audio_path)
        text = model.stt(audio)
        return text
    except Exception as e:
        logging.error(f"Error in transcribe: {e}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            logging.error("No file part in the request")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            logging.error("No selected file")
            return redirect(request.url)
        if file:
            try:
                filename = secure_filename(file.filename)
                input_path = os.path.join('uploads', filename)
                logging.debug(f"Saving uploaded file to {input_path}")
                file.save(input_path)
                logging.debug(f"File saved successfully: {input_path}")

                mp3_path = os.path.join('uploads', 'audio.wav')
                convert_mp4_to_mp3(input_path, mp3_path)

                segments = segment_audio(mp3_path, 'segments')

                transcriptions = [transcribe(segment) for segment in segments]

                return render_template('result.html', transcriptions=transcriptions)
            except Exception as e:
                logging.error(f"Error during processing: {e}")
                return "Internal Server Error", 500
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('segments'):
        os.makedirs('segments')
    app.run(debug=True)
