
# 🎧 Music Genre Classifier & Recommender System

> **A Deep Learning and Signal Processing-powered Web App for Music Genre Classification and Recommendation**

---

## 🚀 Live Demo

🌐 [music-genre-classifier-recommender.streamlit.app](https://music-genre-classifier-recommender.streamlit.app)

> ⚠️ **Note:** The first time launching the app (locally or online) may take **2–3 minutes** as it loads the model, scaler, encoder, and data into memory. This delay only occurs on the initial load; subsequent predictions will be faster.
---

## 📖 Overview

The **Music Genre Classifier & Recommender System** is a Streamlit-based web application that:

- 🎼 **Classifies the genre** of an uploaded `.wav` music clip using a trained **Convolutional Neural Network (CNN)**.
- 🔄 **Recommends 5 similar tracks** based on **audio feature similarity** using **cosine similarity**.

This project combines deep learning and signal processing with an interactive web interface to deliver real-time genre prediction and content-based music recommendations.

---

## ✨ Features

- 🎵 **Genre Classification** for `.wav` audio files.
- 🧠 **CNN Model** trained on GTZAN dataset audio features.
- 🔁 **Top-5 Similar Song Recommendations** using cosine similarity.
- 🎧 **Audio Feature Extraction** including:
  - MFCCs
  - Chroma
  - Spectral Centroid, Rolloff, Bandwidth
  - Zero Crossing Rate
  - Tempo
- 🌐 **Streamlit Web App** with:
  - Genre prediction
  - Audio preview
  - Interactive song recommendations
- 🗒️ **Jupyter Notebooks** for:
  - Training the CNN model
  - Testing and validating the model
  - Feature extraction and preprocessing
- 🚀 Easy deployment via Streamlit Community Cloud.

---

## 🧠 Model Architecture

- **Input:** Audio features extracted from 30-second `.wav` files.
- **Architecture:** Convolutional Neural Network (CNN).
- **Output:** One of 10 genre classes.
- **Prediction Method:** Majority voting over segmented audio chunks for robust accuracy.

---

## 📂 Dataset

- **Dataset:** [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html)
- **Size:** 1000 audio tracks (30 seconds each)
- **Genres:**
  - Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock
- ⚠️ *Note:* The dataset is primarily based on songs from the **1970s to early 2000s**.

---

## 🔧 Tech Stack

| Layer            | Tools/Frameworks                      |
|------------------|----------------------------------------|
| Web App          | Streamlit                             |
| Machine Learning | TensorFlow / Keras                    |
| Feature Extraction| Librosa, NumPy, SciPy                |
| Recommendation   | Scikit-learn (cosine similarity)      |
| Deployment       | Streamlit Community Cloud             |
| Notebook Support | Jupyter, IPython                      |
| Audio Format     | `.wav` (mono, 22050 Hz sample rate)   |

---

## 📁 Folder & File Structure

| File / Folder                       | Description                                              |
|-------------------------------------|----------------------------------------------------------|
| `Music_Genre_App.py`                | Main Streamlit web app                                   |
| `Train_MusicGenre_Classifier.ipynb` | Jupyter Notebook for model training                      |
| `Test_MusicGenreClassifier.ipynb`   | Jupyter Notebook for model testing and validation        |
| `MusicGenreClassifier.keras`        | Trained CNN model                                        |
| `minmax_scaler.pkl`                 | Scaler for feature normalization                         |
| `label_encoder.pkl`                 | Label encoder for genres                                 |
| `gtzan_data.joblib`                 | Preprocessed GTZAN track features for recommendation     |
| `requirements.txt`                  | Required Python packages                                 |
| `README.md`                         | Project documentation                                    |

---

## 🔬 Jupyter Notebooks 📓

### ✅ **Train_MusicGenre_Classifier.ipynb**
- Preprocesses audio features.
- Defines and trains the CNN model.
- Performs feature scaling and label encoding.
- Saves the model (`.keras`), scaler (`.pkl`), and encoder (`.pkl`).

### ✅ **Test_MusicGenreClassifier.ipynb**
- Loads the saved model and scaler.
- Tests genre predictions on sample audio files.
- Evaluates accuracy and performance metrics.
- Can be used for experimentation with different audio clips.

> The notebooks are useful for understanding the complete **ML pipeline** — from preprocessing and training to testing — and for extending or modifying the model if needed.

---

## 💻 Local Installation Guide

### ✅ Requirements

- Python 3.10+
- pip (Python package installer)

### 🔧 Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rishith-Thommandru/music-genre-classifier.git
   cd music-genre-classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run Music_Genre_App.py
   ```

4. **Or open Jupyter notebooks:**
   ```bash
   jupyter notebook
   ```
   - Use the training and testing notebooks for experimenting or retraining the model.

---

## 🔗 Requirements

```txt
streamlit
librosa
scikit-learn
numpy
pandas
joblib
tensorflow
soundfile
scipy
matplotlib
```

---

## 🔥 How It Works

### 🎼 **Genre Classification**
- Splits uploaded audio into segments.
- Extracts audio features (MFCCs, chroma, spectral features, etc.).
- Predicts genre for each segment using the CNN model.
- Uses majority voting for the final prediction.

### 🎧 **Recommendation System**
- Extracts features from uploaded audio.
- Compares with precomputed features from GTZAN dataset.
- Computes cosine similarity and returns the **top 5 most similar tracks**.

---

## 📜 License

This project is for **educational and academic use only.**  
It uses the [GTZAN Genre Collection](https://www.tensorflow.org/datasets/catalog/gtzan) for research purposes.  
All rights to the original audio files belong to their respective owners.

---

## 👨‍💻 Author

- **Rishith Thommandru**  
  [LinkedIn](https://www.linkedin.com/in/rishith-thommandru) • [GitHub](https://github.com/Rishith-Thommandru)

---

## ⭐ Give this repository a ⭐ if you find it helpful!
