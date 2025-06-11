# 😄 Emotion Recognition using CNN

This project is a **machine learning model** built using **Convolutional Neural Networks (CNNs)** that can recognize **human facial emotions** such as *happy, sad, angry, surprised, fearful, disgusted, and neutral* by analyzing facial expressions from images or video frames.

---

## 📌 Features

- 🎯 Built using **CNN architecture** in Keras/TensorFlow.
- 📷 Detects **faces** using OpenCV Haar Cascades.
- 😀 Recognizes **7 core human emotions**.
- 🧠 Trained on grayscale facial images (48x48).
- 🔍 Works with uploaded **images or video files**.
- 🧪 Real-time prediction code included, but limited in Colab.

---

## 🛠 Tools & Libraries

- Python
- Keras + TensorFlow
- OpenCV
- NumPy, Matplotlib
- Google Colab (for development)

---

## 🚀 How It Works

1. A video or image is uploaded to the Google Colab notebook.
2. The face is detected using Haar cascades (`haarcascade_frontalface_default.xml`).
3. Each detected face is resized to **48x48 grayscale**.
4. The CNN model predicts the **emotion label** for the face.
5. Output is shown with **emotion tags** drawn on each face.

---

## ⚡ Real-Time Capability

> While the model supports real-time webcam input,
> **Google Colab cannot access the webcam stream in real time** due to browser security restrictions.

### ✅ Workaround:
- A **JavaScript-based image capture** is used in Colab.
- Captured frames are processed individually to simulate real-time prediction.
- Video file upload support added for frame-by-frame emotion detection.

---

## 🎓 Emotions Recognized

- 😡 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

---

## 📂 Files in This Project

| File                      | Purpose                                         |
|---------------------------|--------------------------------------------------|
| `emotiondetector.json`    | CNN model architecture                         |
| `emotiondetector.h5`      | Trained model weights                          |
| `haarcascade_frontalface_default.xml` | Face detection cascade file           |
| `emotion_colab_notebook.ipynb` | Main notebook for Colab-based demo         |
| `README.md`               | This project description                       |

---

## 📌 Future Enhancements

- Save processed video with emotions labeled.
- Deploy to web using Flask + JavaScript.
- Improve accuracy using more robust datasets.

---
