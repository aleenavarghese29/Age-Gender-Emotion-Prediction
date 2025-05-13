import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Path to UTKFace dataset
dataset_path = 'utkface_dataset/UTKFace'

# Data preprocessing with checks
def preprocess_data():
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset directory does not exist: {dataset_path}")
    
    images = []
    labels_age = []
    labels_gender = []

    # Iterate through the images in the dataset
    for img_name in os.listdir(dataset_path):
        print(f"Processing {img_name}")
        if img_name.endswith('.jpg'):
            img_path = os.path.join(dataset_path, img_name)
            print(f"Processing {img_path}")  # Debugging line to ensure files are found
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img) / 255.0
            images.append(img)

            # Extract age and gender from filename
            try:
                age, gender, _ = img_name.split('_')[:3]
                labels_age.append(int(age))
                labels_gender.append(int(gender))
            except ValueError:
                print(f"Skipping invalid filename: {img_name}")
                continue

    if len(images) == 0:
        raise ValueError("No images found in the dataset directory.")
    images = np.array(images)
    labels_age = np.array(labels_age)
    labels_gender = np.array(labels_gender)

    # Check if labels have the expected number of classes
    if not np.all(np.isin(labels_gender, [0, 1])):
        raise ValueError("Gender labels must be 0 or 1.")

    labels_gender = to_categorical(labels_gender, 2)

    X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender = train_test_split(
        images, labels_age, labels_gender, test_size=0.2, random_state=42)

    return X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender

# Model for age and gender prediction
def create_model():
    input_layer = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    gender_output = Dense(2, activation='softmax', name='gender_output')(x)
    age_output = Dense(1, activation='linear', name='age_output')(x)

    model = Model(inputs=input_layer, outputs=[gender_output, age_output])

    model.compile(
        optimizer='adam',
        loss={
            'gender_output': 'categorical_crossentropy',
            'age_output': 'mean_squared_error'
        },
        metrics={
            'gender_output': ['accuracy'],
            'age_output': ['mae']
        }
    )

    return model

# Train the model with additional checks
def train_model():
    try:
        X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender = preprocess_data()
        model = create_model()

        history = model.fit(
            X_train,
            {'gender_output': y_train_gender, 'age_output': y_train_age},
            epochs=10,
            batch_size=32,
            validation_data=(X_test, {'gender_output': y_test_gender, 'age_output': y_test_age})
        )

        model.save('age_gender_model.h5')

        # Plot training metrics
        plt.plot(history.history['gender_output_accuracy'], label='Gender Accuracy')
        plt.plot(history.history['age_output_mae'], label='Age MAE')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    train_model()