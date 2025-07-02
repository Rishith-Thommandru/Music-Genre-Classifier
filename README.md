# 🎧 Music Genre Classifier & Recommender System

> **A Deep Learning and Signal Processing-based Web App for Audio Genre Classification and Recommendation**

---

## 🔗 Live App

> 🌍 [music-genre-classifier-recommender.streamlit.app](https://music-genre-classifier-recommender.streamlit.app)

---

## 📚 Project Overview

The **Music Genre Classifier & Recommender** is a Streamlit web application that classifies the genre of an uploaded music clip using a CNN model trained on audio features, and recommends similar tracks based on feature similarity.

Built using the **GTZAN genre dataset**, the app combines deep learning for genre prediction with cosine similarity for music recommendation — all powered by real-time audio signal analysis.

---

## 🚀 Features

- 🎼 Genre Classification of uploaded `.wav` music files
- 🧠 Trained CNN Model using GTZAN audio features
- 🔁 Top-5 Music Recommendations using cosine similarity
- 🎧 Deep Audio Feature Extraction using Librosa (MFCCs, Chroma, ZCR, Tempo, etc.)
- 📊 Preprocessed GTZAN Track Database for similarity search
- 🌐 Streamlit Web App with interactive audio preview and predictions
- 📦 Easily deployable via Streamlit Community Cloud

---


## 🧠 Model Summary

- Input: Extracted features from 30s audio clips
- Architecture: Convolutional Neural Network (CNN)
- Output: Predicted genre from 10 GTZAN classes
- Prediction: Based on majority voting across segmented audio

---

## 🗂️ Dataset Used

- **Dataset**: [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html)
- **Samples**: 1000 audio tracks
- **Duration**: Each clip is 30 seconds
- **Genres**:  blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock


---

## 🛠️ Technologies Used

| Layer         | Tools/Frameworks                        |
|---------------|------------------------------------------|
| Web App       | Streamlit                               |
| ML Framework  | TensorFlow / Keras                      |
| Feature Eng.  | Librosa, NumPy, SciPy                   |
| Recommendation| Scikit-learn (cosine similarity)        |
| Deployment    | Streamlit Community Cloud               |
| Audio Files   | `.wav` format (mono, 22050 Hz expected) |

---


## 📂 Project Structure

| File / Folder                  | Description                                            |
|-------------------------------|---------------------------------------------------------|
| `Music_Genre_App.py`          | Main Streamlit app script                               |
| `requirements.txt`            | Required packages                                       |
| `runtime.txt`                 | Python version for deployment                           |
| `model/`                      | Folder containing trained model and preprocessing tools |
| `MusicGenreClassifier.keras`  | Trained CNN model                                       |
| `minmax_scaler.pkl`           | Scaler for feature normalization                        |
| `label_encoder.pkl`           | Label encoder for genres                                |
| `data/gtzan_data.joblib`      | Preprocessed GTZAN track features for recommendation    |
| `Input/genres_original/`      | GTZAN dataset audio clips (used for recommendations)    |
| `README.md`                   | Project documentation                                   |

---


## 🏗️ How to Run (locally)

1. Install Python 3.10+ and ensure `pip` is available.
2. Clone the repository:
   ```bash
   git clone https://github.com/Rishith-Thommandru/music-genre-classifier
   cd music-genre-classifier
3. Install the required dependencies
   ```bash
   pip install -r requirements.txt
4. Run streamlit app
   ```bash
   streamlit run Music_Genre_App.py

---

## 👨‍💻 Authors

- **Rishith Thommandru** — [LinkedIn](https://www.linkedin.com/in/rishith-thommandru)

---

## 📜 License

This project is developed for educational and academic demonstration purposes only.  
It uses the [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html), which is publicly available for research and academic use.  
All rights to the original audio content belong to their respective owners.

---
