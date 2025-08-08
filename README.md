# 🎧 Music Genre Classifier & Recommender System

> **A Deep Learning & Signal Processing–powered Web App for Music Genre Classification and Recommendation**

---

## 🚀 Live Demo

🌐 [https://music-genre-classifier-recommender.streamlit.app](https://music-genre-classifier-recommender.streamlit.app)

> ⚠️ **Note:** The first time launching the app (locally or online) may take **2–3 minutes** as it loads the model, scaler, encoder, and data into memory. This delay only occurs on the initial load; subsequent predictions will be faster.

---

## 📖 Overview

The **Music Genre Classifier & Recommender System** is a Streamlit-based web application that:

* 🎼 **Classifies the genre** of an uploaded `.wav` music clip using a trained **Fully Connected Neural Network (MLP)**.
* 🔄 **Recommends 5 similar tracks** based on **audio feature similarity** using **cosine similarity**.

This project combines deep learning with audio signal processing and an interactive web interface to deliver real-time genre prediction and content-based music recommendations.

---

## ✨ Features

* 🎵 **Genre Classification** for `.wav` audio files.
* 🧠 **MLP Model** trained on GTZAN dataset audio features extracted via Librosa.
* 🔁 **Top-5 Similar Song Recommendations** using cosine similarity.
* 🎧 **Audio Feature Extraction** including:

  * MFCCs (mean & variance)
  * Chroma features
  * Spectral Centroid, Rolloff, Bandwidth
  * Zero Crossing Rate
  * Tempo
* 🌐 **Streamlit Web App** with:

  * Genre prediction
  * Audio preview
  * Interactive song recommendations
* 🗒️ **Jupyter Notebooks** for:

  * Training the MLP model
  * Testing and validating the model
  * Feature extraction and preprocessing
* 🚀 Easy deployment via Streamlit Community Cloud.

---

## 🧠 Model Architecture

* **Input:** 58 precomputed audio features extracted from \~3-second segments of `.wav` files.
* **Architecture:** Fully Connected Neural Network (Dense layers with ReLU + Dropout).
* **Output:** One of 10 genre classes.
* **Prediction Method:** Majority voting over segmented audio chunks for robust accuracy.

---

## 📂 Dataset

* **Dataset:** GTZAN Genre Collection
* **Size:** 1000 audio tracks (30 seconds each)
* **Genres:** Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock
* ⚠️ *Note:* The dataset is primarily based on songs from the **1970s to early 2000s**.

---

## 🔧 Tech Stack

| Layer              | Tools/Frameworks                    |
| ------------------ | ----------------------------------- |
| Web App            | Streamlit                           |
| Machine Learning   | TensorFlow / Keras (MLP)            |
| Feature Extraction | Librosa, NumPy, SciPy               |
| Recommendation     | Scikit-learn (cosine similarity)    |
| Deployment         | Streamlit Community Cloud           |
| Notebook Support   | Jupyter, IPython                    |
| Audio Format       | `.wav` (mono, 22050 Hz sample rate) |

---

## 📁 Folder & File Structure

| File / Folder                       | Description                                          |
| ----------------------------------- | ---------------------------------------------------- |
| `Music_Genre_App.py`                | Main Streamlit web app                               |
| `Train_MusicGenre_Classifier.ipynb` | Jupyter Notebook for model training                  |
| `Test_MusicGenreClassifier.ipynb`   | Jupyter Notebook for model testing and validation    |
| `MusicGenreClassifier.keras`        | Trained MLP model                                    |
| `minmax_scaler.pkl`                 | Scaler for feature normalization                     |
| `label_encoder.pkl`                 | Label encoder for genres                             |
| `gtzan_data.joblib`                 | Preprocessed GTZAN track features for recommendation |
| `requirements.txt`                  | Required Python packages                             |
| `README.md`                         | Project documentation                                |

---

## 🔬 Jupyter Notebooks 📓

### ✅ **Train\_MusicGenre\_Classifier.ipynb**

* Loads and preprocesses extracted audio features.
* Scales features and encodes labels.
* Defines and trains the MLP model (Dense + Dropout layers).
* Saves the model (`.keras`), scaler (`.pkl`), and encoder (`.pkl`).

### ✅ **Test\_MusicGenreClassifier.ipynb**

* Loads the saved model, scaler, and label encoder.
* Accepts a `.wav` file as input.
* Automatically segments audio, extracts features, scales them, predicts genre per segment, and uses majority voting for final output.
* Evaluates accuracy and plots confusion matrix.

---

## 💻 Local Installation Guide

### ✅ Requirements

* Python 3.10+
* pip (Python package installer)

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

* Splits uploaded audio into \~3-second segments.
* Extracts 58 statistical audio features per segment using Librosa.
* Scales features with a saved MinMaxScaler.
* Predicts genre for each segment using the trained MLP model.
* Uses majority voting for the final prediction.

### 🎧 **Recommendation System**

* Extracts features from the uploaded audio.
* Compares with precomputed GTZAN features.
* Computes cosine similarity and returns the **top 5 most similar tracks**.

---

## 🚀 Future Improvements

* 🔗 Convert to a **true ML end-to-end pipeline** using spectrograms + CNN/RNN so feature extraction is learned by the model.
* 🔗 Integrate with the **YouTube API** for dynamic song fetching.
* 🗣️ Expand genre coverage using larger and more diverse datasets.
* 📱 Convert into a full-stack web application with user accounts and playlists.
* ⚙️ Explore deployment on scalable cloud platforms (AWS, Azure, GCP).

---

## 📜 License

This project is for **educational and academic use only.**
It uses the GTZAN Genre Collection for research purposes.
All rights to the original audio files belong to their respective owners.

---

## 👨‍💻 Author

* **Rishith Thommandru**
  [LinkedIn](https://www.linkedin.com/in/rishith-thommandru) • [GitHub](https://github.com/Rishith-Thommandru)

---

## ⭐ Thanks

If you'd like, I can also:

* Add a simple architecture diagram (MLP vs CNN) as an image and update README.
* Provide a `README` variant that emphasizes deployment instructions for Streamlit Cloud.
* Generate a `requirements.txt` tuned to exact package versions used in your notebooks.
