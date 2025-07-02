import streamlit as st
import librosa
import pandas as pd
import numpy as np
from scipy.stats import mode
import joblib
import random


from tensorflow.keras.models import load_model

from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

import soundfile as sf
from io import BytesIO

@st.cache_resource()        # to avoid loading multiple times, it will be stored in cache
def load_model_and_scaler_encoder():
    # Load trained model
    model = load_model("MusicGenreClassifier.keras")

    # Load scaler and label encoder
    min_max_scaler = joblib.load("minmax_scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    return model,min_max_scaler,label_encoder

def split_audio(file_path, sr=22050, segment_samples=66149, max_segments=12):
    y, _ = librosa.load(file_path, sr=sr)  # Force resampling to 22050 Hz

    # Pad to make length divisible by segment_samples
    remainder = len(y) % segment_samples
    if remainder != 0:
        pad_width = segment_samples - remainder
        y = np.pad(y, (0, pad_width))

    segments = [y[i:i+segment_samples] for i in range(0, len(y), segment_samples)]

    # Randomly sample up to max_segments
    if len(segments) <= max_segments:
        return segments
    else:
        return random.sample(segments, max_segments)

def extract_full_features(y, sr=22050):
    features = {}

    # Length (number of samples)
    features['length'] = len(y)

    # Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_stft_mean'] = np.mean(chroma)
    features['chroma_stft_var'] = np.var(chroma)

    # RMS
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)

    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(cent)
    features['spectral_centroid_var'] = np.var(cent)

    # Spectral Bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(bw)
    features['spectral_bandwidth_var'] = np.var(bw)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_var'] = np.var(rolloff)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_var'] = np.var(zcr)

    # Harmonic/Perceptual
    harmony = librosa.effects.harmonic(y)
    perceptr = librosa.effects.percussive(y)
    features['harmony_mean'] = np.mean(harmony)
    features['harmony_var'] = np.var(harmony)
    features['perceptr_mean'] = np.mean(perceptr)
    features['perceptr_var'] = np.var(perceptr)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfcc[i-1])
        features[f'mfcc{i}_var'] = np.var(mfcc[i-1])

    return features

def process_and_predict(song_path):
    segments = split_audio(song_path)
    all_preds = []

    model,min_max_scaler,label_encoder=load_model_and_scaler_encoder()

    for seg in segments:
        fvec = extract_full_features(seg)
        df = pd.DataFrame([fvec])
        X_scaled = min_max_scaler.transform(df)
    
        probs = model.predict(X_scaled)
        pred_class = np.argmax(probs, axis=1)[0]
        all_preds.append(pred_class)

# Majority vote
    majority_class = mode(all_preds, keepdims=False).mode
    predicted_genre = label_encoder.inverse_transform([majority_class])[0]

    return predicted_genre

@st.cache_data
def load_gtzan_data():
    data = joblib.load('gtzan_data.joblib')
    return data['gtzan_scaled'], data['gtzan_filenames'], data['scaler']

def recommend_similar_songs(audio_path, gtzan_scaled, gtzan_filenames, scaler, top_n=5):
    y, sr = librosa.load(audio_path, sr=22050, duration=30.0)

    input_features = extract_full_features(y, sr)
    df_input = pd.DataFrame([input_features])
    df_input = pd.DataFrame([input_features]).drop(columns=['length'])
    input_scaled = scaler.transform(df_input)

    similarities = cosine_similarity(input_scaled, gtzan_scaled).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    return [(gtzan_filenames[i], similarities[i]) for i in top_indices]

def recommend_top5_songs(song_path):
    gtzan_scaled, gtzan_filenames, scaler=load_gtzan_data()
    recommendations = recommend_similar_songs(
        song_path,
        gtzan_scaled,
        gtzan_filenames,
        scaler,
        top_n=5
    )
    return recommendations




st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Prediction", "Explore Similar Tracks"])

st.title("🎧 Music Genre Classification & Recommendation")



if(app_mode=="Home"):
    st.markdown("""
    ### 🎶 Welcome to the Music Genre Classifier App!

    Upload your favorite music and find out its **genre** instantly using deep learning.  
    Then get **smart recommendations** based on musical similarity using deep audio features.
    

    ---

    ### 📚 About the GTZAN Dataset

    This app is powered by the **GTZAN genre dataset**, which contains:
    - **1,000 audio tracks**, each 30 seconds long
    - Spread across **10 genres**: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
    - Most tracks are from songs **released between 1970 and 2000**

    > ⚠️ For best prediction and recommendation accuracy, we suggest uploading `.wav` clips from music released **during or before the early 2000s**, as they closely match the training data distribution.

    ---

    ### 🚀 Features
    - 🎵 Genre classification using deep learning
    - 🤖 Song recommendations based on cosine similarity of audio features
    - 📊 Feature extraction using MFCCs, chroma, spectral, and rhythmic characteristics

    ---

    ### 🔗 More Info & Source Code

    Check out the [GitHub repository and README](https://github.com/your-username/your-repo-name) for:
    - 📁 Dataset details
    - 🧠 Model architecture
    - 🔧 Setup instructions and dependencies

    ---

    Navigate to **Prediction** or **Recommender** from the sidebar to get started!
    """)



elif app_mode == "Prediction":
    st.header("🎼 Genre Prediction")
    st.markdown("""
    Upload a `.wav` file and get an instant genre prediction using a trained Convolutional Neural Network (CNN).

    > 📌 **Note:** The model was trained on the **GTZAN dataset**, which consists of songs mostly from the **1970s to early 2000s**. For best results, upload music from a similar era.
    """)


    uploaded_file = st.file_uploader("🎵 Upload your WAV file below:", type=["wav"])

    audio_buffer = None

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        audio_buffer = BytesIO(audio_bytes)
        st.audio(audio_buffer, format='audio/wav')


    
    if st.button("🎯 Predict Genre"):
        if audio_buffer is None:
            st.error("Please upload a song first.")
        else:
            with st.spinner("Analyzing features and predicting genre..."):
                st.info("⏳ Loading model and encoders... ⚠️ This may take **1–2 minutes on first prediction**.")
                predicted_genre = process_and_predict(audio_buffer)
            st.success(f"**Predicted Genre:** {predicted_genre}")
            st.balloons()



elif app_mode == "Explore Similar Tracks":
    st.header("🔍 Explore Similar Tracks")

    st.markdown("""
    Upload a `.wav` file to get 5 similar songs based on audio content, using cosine similarity on extracted features.

    > ⚠️ **Note:** Audio files from the GTZAN dataset are not included in this demo due to size and licensing restrictions.  
    > Only filenames and similarity scores are shown here.  
    > Audio playback will work **only if the GTZAN dataset is available locally** in the folder: `Dataset/genres_original`.

    > 🎧 **Want to try it with your own songs and GTZAN audio samples?**  
    👉 Check out the full version here:  
    [Kaggle Notebook – Recommender System](https://www.kaggle.com/code/darthchaos/recommender-system)
    """)

    uploaded_file = st.file_uploader("🎵 Upload a WAV audio file", type=["wav"])
    audio_buffer = None

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        audio_buffer = BytesIO(audio_bytes)
        st.audio(audio_buffer, format='audio/wav')

    if st.button("🔁 Recommend 5 Similar Songs"):
        if audio_buffer is None:
            st.error("Please upload a song first.")
        else:
            with st.spinner("Analyzing and finding similar tracks..."):
                recommendations = recommend_top5_songs(audio_buffer)

            st.subheader("🎧 Top 5 Recommended Tracks")
            for i, (fname, sim) in enumerate(recommendations, 1):
                st.markdown(f"**{i}. {fname}**      *Similarity score:* `{sim:.2f}`")

                genre_folder = fname.split(".")[0]
                audio_file_path = f"Dataset/genres_original/{genre_folder}/{fname}"

                try:
                    with open(audio_file_path, 'rb') as f:
                        audio_data = f.read()
                        st.audio(audio_data, format='audio/wav')
                except FileNotFoundError:
                    st.info("🔇 Audio sample not available locally.")


