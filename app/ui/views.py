import os
import cv2
import numpy as np
import time
import librosa
import librosa.display
import pickle

import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from flask import jsonify

import scipy.io.wavfile as wav
from scipy.signal import medfilt, butter, filtfilt
from scipy import signal

from moviepy.editor import VideoFileClip
from flask import Blueprint, render_template, request, redirect, url_for , session
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions

from pydub import AudioSegment

import numpy as np
import json

ui_blueprint = Blueprint('ui', __name__)

AUDIO_FILE_PATH = 'app/bird_audio.wav'

with open('app/models/class_indices.json', 'r') as f:
    class_indices = json.load(f)

with open('app/models/image_Model_class.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the class indices mapping
class_indices_inv = {v: k for k, v in class_indices.items()}

# Load the trained model
# model = tf.keras.models.load_model('app/models/bird_species_resnet50v2.h5')
model = tf.keras.models.load_model('app/models/Bird_image_model.h5')

audiomodel = tf.keras.models.load_model('app/models/birds_classifier_model.h5')

label_encoder = LabelEncoder()
# label_encoder.classes_ = np.load('app/ui/label_encoder.npy', allow_pickle=True)
# Define the folder containing the images of birds

with open('app/ui/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)



folder_path = 'app/bird_frames/'

audio_folder_path = 'app/output_audio/'

@ui_blueprint.route('/')
def index():
    return render_template('index.html')

# @ui_blueprint.route('/home')
# def index():
#     return render_template('home.html')
@ui_blueprint.route('/get_second_image')
def get_second_image():
    # Assuming bird_frames is a folder in the same directory as your Flask app
    folder_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'bird_frames')

    # List all files in the folder
    files = os.listdir(folder_path)

    # Assuming the second image is the one at index 1
    if len(files) >= 2:
        image_url = f'/bird_frames/{files[1]}'  # Construct the URL to the second image
        return jsonify({'image_url': image_url})
    else:
        return jsonify({'image_url': None})

@ui_blueprint.route('/results')
def show_results():
    predicted_species = request.args.get('predicted_species', '')
    return render_template('results.html', predicted_species=predicted_species)



@ui_blueprint.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video_file' not in request.files:
        return redirect(request.url)
    
    file = request.files['video_file']
    if file.filename == '':
        return redirect(request.url)
    
    # Delete existing images in the bird_frames folder
    existing_frames_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'bird_frames')
    for filename in os.listdir(existing_frames_path):
        file_path = os.path.join(existing_frames_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Save the uploaded video to a temporary location
    uploaded_video_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'uploads', file.filename)
    file.save(uploaded_video_path)
    
    # Open the uploaded video file with OpenCV
    cap = cv2.VideoCapture(uploaded_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the desired frames per second (e.g., 1 fps)
    desired_fps = 1

    # Calculate the frame interval to achieve the desired fps
    frame_interval = int(round(fps / desired_fps))

    # Open the uploaded video file with moviepy
    clip = VideoFileClip(uploaded_video_path)

    # Create directories for frames and audio if they don't exist
    frames_output_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'bird_frames')
    os.makedirs(frames_output_path, exist_ok=True)
    audio_output_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'output_audio', 'bird_audio.mp3')

    # audio_output_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'bird_audio.wav')

    # Frame extraction
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract one frame per second
        if frame_count % frame_interval == 0:
            resized_frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(os.path.join(frames_output_path, f'frame_{frame_count // frame_interval + 1}.png'), resized_frame)

        frame_count += 1

    # Audio extraction
    audio = clip.audio
    audio.write_audiofile(audio_output_path,codec='mp3')

    # Release the video capture object
    cap.release()

    # Close the clip
    clip.close()
    return redirect(url_for('.index'))



@ui_blueprint.route('/preprocess_audio', methods=['GET'])  # Change to GET method
def preprocess_audio():
    # Path to the bird.wav file in the root directory
    audio_path = 'app/bird_audio.wav'
    output_path = 'app/output_audio.wav'
    
    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Apply noise reduction using a larger median filtering kernel size
    y_denoised = medfilt(y, kernel_size=5)  # Increase kernel size to 5
    
    # Resample the audio to a common sample rate if needed (e.g., 16kHz)
    target_sr = 16000
    if sr != target_sr:
        y_resampled = signal.resample(y_denoised, int(len(y_denoised) * target_sr / sr))
        sr = target_sr
    else:
        y_resampled = y_denoised
    
    # Apply normalization to the audio
    y_normalized = librosa.util.normalize(y_resampled)
    
    # Apply a high-pass filter to remove low-frequency noise
    y_high_pass = librosa.effects.preemphasis(y_normalized)
    
    # Apply a low-pass filter with adjusted parameters
    cutoff_frequency = 7000  # Adjust cutoff frequency to 7000 Hz
    b, a = butter(4, cutoff_frequency / (sr / 2), 'low')
    y_low_pass = filtfilt(b, a, y_high_pass)
    
    output_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'output_audio')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'preprocessed_audio_{int(time.time())}.wav')
    wavfile.write(output_path, sr, (y_low_pass * 32767).astype(np.int16))

    # Spectrogram analysis
    D_original = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    D_processed = librosa.amplitude_to_db(np.abs(librosa.stft(y_low_pass)), ref=np.max)

    # Plot the original and preprocessed audio spectrogram
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(D_original, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of Original Audio')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(D_processed, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of Preprocessed Audio')

    plt.tight_layout()
    plt.show()

    # Optionally, you can redirect the user to a page to display a success message or perform further actions
    return redirect(url_for('.index')) # Redirect to the index route within the same blueprint
 # Redirect to the index route within the same blueprint



@ui_blueprint.route('/predict_species', methods=['GET'])
def predict_species():
    # Initialize variables to store the predicted species and maximum score
    predicted_species = None
    max_score = 0.0

    # Iterate over each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
            # Load and preprocess the image
            img_path = os.path.join(folder_path, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Make predictions
            predictions = model.predict(img_array)

            # Get the top predicted class and score
            top_class_index = np.argmax(predictions)
            top_class = class_indices_inv[top_class_index]
            top_score = predictions[0][top_class_index]

            # If predicted_species is not set, or if the current species has higher probability, update predicted_species
            if predicted_species is None or top_score > max_score:
                predicted_species = top_class
                max_score = top_score

    
            print(predicted_species)

    print(max_score)

    # session['species_from_images'] = predicted_species
    # if 'species_from_audio' in session:
    #     return redirect(url_for('.results'))


    # return redirect(url_for('.index'))

    # Return the predicted species as JSON response
    return jsonify({"predicted_species": predicted_species , "probabilites":float(max_score)})

# def preprocess_audio(audio_path):
#     SR = 16000  # Sample rate
#     DURATION = 5  # Duration of audio clips in seconds
#     audio, _ = librosa.load(audio_path, sr=SR, duration=DURATION, mono=True)
#     # Convert audio to spectrogram
#     spectrogram = librosa.feature.melspectrogram(y=audio, sr=SR)
#     return np.expand_dims(spectrogram, axis=0)


# def predict_bird_species(audio_file):
#     # Convert MP3 to WAV
    
#     # Preprocess input audio file
#     processed_input = preprocess_audio(audio_file)
#     # Make prediction
#     predictions = model.predict(processed_input)
#     # Decode predicted label
#     predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])[0]
#     return predicted_label

# @ui_blueprint.route('/predict_species_audio', methods=['GET'])
# def predict_species_audio():
#     predicted_species = []

#     # List all audio files in the directory
#     audio_files = os.listdir(audio_folder_path)

#     # Predict species for each audio file
#     for audio_file in audio_files:
#         audio_file_path = os.path.join(audio_folder_path, audio_file)
#         predicted_species.append(predict_bird_species(audio_file_path))

#     # Return the predicted species as JSON response
#     return jsonify({"predicted_species": predicted_species})

@ui_blueprint.route('/predict_species_audio', methods=['GET'])
def predict_species_audio():
    predicted_species = []

    # List all audio files in the directory
    audio_folder_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), 'output_audio')
    audio_files = [f for f in os.listdir(audio_folder_path) if f.endswith('.mp3')]
    if not audio_files:
        return jsonify({"error": "No audio file found in output_audio folder"})

    audio_file_path = os.path.join(audio_folder_path, audio_files[0])
    # print("Audio Files : " ,audio_files[0])
    # Predict species for each audio file
    predicted_label = predict_audio_label(audio_file_path)
    print("Predicted label:", predicted_label)

    # Return the predicted species as JSON response
    return jsonify({"predicted_species":predicted_label})

# Function to load audio file and convert to spectrogram
def load_audio(file_path):
    SR = 16000  # Sample rate
    DURATION = 5  # Duration of audio clips in seconds
    audio, _ = librosa.load(file_path, sr=SR, duration=DURATION, mono=True)
    # Convert audio to spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=SR)
    return spectrogram

# Load the trained model

# Load the label encoder


# Function to predict label of an audio file
def predict_audio_label(audio_file):
    # Load and preprocess the audio file
    spectrogram = load_audio(audio_file)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension

    # Predict probabilities for each class
    predicted_probabilities = audiomodel.predict(spectrogram)
    
    # Get the predicted class index
    predicted_index = np.argmax(predicted_probabilities)
    
    
    # Map the index to the corresponding class label
    predicted_label = list(label_encoder.keys())[predicted_index]
    predicted_probability = predicted_probabilities[0][predicted_index]
    print(predicted_probability)

    # session['species_from_audio'] = predicted_label
    # if 'species_from_images' in session:
    #     return redirect(url_for('.results'))

    # return redirect(url_for('.index'))
    return predicted_label

# Example usage
