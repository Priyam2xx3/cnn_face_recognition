# 📌 Step 1: Install & Import Required Libraries
from IPython.display import display, Javascript
from google.colab.output import eval_js
import cv2
import numpy as np
from keras.models import model_from_json
from base64 import b64decode
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import io

# 📌 Step 2: JavaScript for Auto Capture (no button click)
def auto_capture_photo(filename='auto_photo.jpg', quality=0.8):
    js = Javascript('''
        async function autoCapturePhoto(quality) {
          const video = document.createElement('video');
          video.style.display = 'block';
          const stream = await navigator.mediaDevices.getUserMedia({video: true});
          document.body.appendChild(video);
          video.srcObject = stream;
          await video.play();

          // Wait 2 seconds for camera to adjust
          await new Promise(resolve => setTimeout(resolve, 2000));

          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext('2d').drawImage(video, 0, 0);
          stream.getVideoTracks()[0].stop();
          video.remove();
          return canvas.toDataURL('image/jpeg', quality);
        }
    ''')
    display(js)
    data = eval_js(f'autoCapturePhoto({quality})')
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# 📌 Step 3: Load the CNN Model
with open("emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("emotiondetector.h5")

# 📌 Step 4: Setup Face Detection and Emotion Labels
labels = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise'
}
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def extract_features(image):
    image = np.array(image).reshape(1, 48, 48, 1)
    return image / 255.0

# 📌 Step 5: Emotion Prediction Function
def predict_emotion(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        processed = extract_features(face)
        prediction = model.predict(processed, verbose=0)
        emotion = labels[np.argmax(prediction)]

        print(f"Detected Emotion: {emotion}")
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img

# 📌 Step 6: Automatically Capture & Predict Emotion
filename = auto_capture_photo()
result_img = predict_emotion(filename)

# 📌 Step 7: Display Output
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Auto-Captured Emotion Detection")
plt.show()
