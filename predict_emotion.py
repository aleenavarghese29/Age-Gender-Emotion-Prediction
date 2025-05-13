import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained emotion model
emotion_model = tf.keras.models.load_model('emotion_model.h5')

# Class labels 
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load and preprocess image
img_path = 'G:/kaggle/fer2013_dataset/test/sad/PrivateTest_528072.jpg'
img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize

# Predict
prediction = emotion_model.predict(img_array)
predicted_class = np.argmax(prediction)

# Result
print(f"Predicted Emotion: {class_labels[predicted_class]}")
