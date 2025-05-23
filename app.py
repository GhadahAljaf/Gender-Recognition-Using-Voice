from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import tensorflow as tf
import joblib
from io import BytesIO
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"

# Load model and scaler
model = tf.keras.models.load_model("model-final.keras")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return jsonify({"message": "API running"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        if not file.filename.lower().endswith('.wav'):
            file = convert_to_wav(file)

        features = preprocess_audio(file)
        prediction = model.predict(features)
        label = 'Female' if prediction[0][0] > 0.5 else 'Male'
        confidence = float(prediction[0][0]) if label == 'Female' else 1 - float(prediction[0][0])

        return jsonify({
            'prediction': label,
            'probability': round(confidence, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_to_wav(file):
    filename = file.filename
    file.seek(0)
    audio_data = file.read()
    audio_stream = BytesIO(audio_data)

    if filename.lower().endswith(".mp3"):
        sound = AudioSegment.from_file(audio_stream, format="mp3")
    elif filename.lower().endswith(".webm"):
        sound = AudioSegment.from_file(audio_stream, format="webm")
    else:
        file.seek(0)
        return file

    wav_io = BytesIO()
    sound.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

def preprocess_audio(file):
    y, sr = librosa.load(file, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=20)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return scaler.transform(mfcc_mean.reshape(1, -1))

if __name__ == '__main__':
    app.run(debug=True)
