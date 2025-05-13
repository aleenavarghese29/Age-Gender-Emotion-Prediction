# Age Gender Emotion Prediction from Facial Images

This project applies deep learning techniques to predict **age**, **gender**, and **emotion** from facial images using the **UTKFace** and **FER2013** datasets.

---

## 📁 Project Structure

- `train_age_gender.py` – Trains the CNN model for age and gender prediction using UTKFace.
- `train_emotion.py` – Trains the CNN model for emotion recognition using FER2013.
- `predict_age_gender.py` – Loads model and predicts age/gender from input image.
- `predict_emotion.py` – Loads model and predicts emotion from input image.
- `fer2013_dataset/` – Contains data for emotion recognition.
- `utkface_dataset/` – Contains data for age and gender prediction.
- `emotion_model.h5`, `age_gender_model.h5` – Trained model weights.

---

## 🧠 Technologies Used

- Python 3.x
- TensorFlow / Keras
- Matplotlib
- NumPy & Pandas
- Scikit-learn

---

## 📈 Results

- Accuracy and loss plots for both models.
- Age model evaluated using Mean Absolute Error (MAE).
- Emotion model evaluated using accuracy and confusion matrix.



---

## 🚀 How to Run

1. **Clone the Repository**

   ```bash
   git clone https://github.com/aleenavarghese29/Age-Gender-Emotion-Prediction.git
   cd Age-Gender-Emotion-Prediction
