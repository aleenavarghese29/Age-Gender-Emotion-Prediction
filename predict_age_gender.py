import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained age-gender model
age_gender_model = tf.keras.models.load_model('age_gender_model.h5')

# Load and preprocess image
img_path = 'fer2013_dataset/test/happy/PrivateTest_647018.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize

# Predict
gender_pred, age_pred = age_gender_model.predict(img_array)

# Get gender result
gender_label = 'Male' if np.argmax(gender_pred) == 1 else 'Female'

# Get age result

predicted_age = int(age_pred[0][0])

# Result
print(f"Predicted Gender: {gender_label}")
print(f"Predicted Age: {predicted_age}")
