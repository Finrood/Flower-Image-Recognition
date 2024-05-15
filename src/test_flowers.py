import os
from pathlib import Path

import numpy as np
from imageio import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Constants
TEST_DIR = Path("datasets/test_flowers")
MODEL_PATH = Path("models/Flower-test-10f.model")
IMG_SIZE = 128
CATEGORIES = ['daisy', 'dandelion', 'Gerbera', 'Iris', 'jonquille', 'Lilac', 'Orchid', 'rose', 'sunflower', 'tulip']
VALID_EXTENSIONS = {'jpg', 'tif', 'png', 'bmp'}


# Load and preprocess the test data
def load_test_data(test_dir, img_size, valid_extensions):
    image_data = []
    image_labels = []

    for category_dir in test_dir.iterdir():
        if category_dir.is_dir():
            category = category_dir.name
            print(f"Processing... {category}")
            for img_path in category_dir.iterdir():
                if img_path.suffix[1:] in valid_extensions:
                    img = imread(img_path)
                    resized_img = resize(img, (img_size, img_size), anti_aliasing=True)
                    image_data.append(resized_img)
                    image_labels.append(category)

    print("Data loading DONE!")
    return np.array(image_data), np.array(image_labels)


# Preprocess the data
def preprocess_data(image_data, image_labels, categories):
    image_data = image_data / 255.0
    label_encoder = LabelEncoder()
    image_labels_encoded = label_encoder.fit_transform(image_labels)
    image_labels_one_hot = to_categorical(image_labels_encoded, num_classes=len(categories))
    return image_data, image_labels_one_hot, label_encoder


# Evaluate the model
def evaluate_model(model, X_test, y_test, categories):
    right_answers = 0
    wrong_answers = 0

    for index, test_img in enumerate(X_test):
        test_img_expanded = np.expand_dims(test_img, axis=0)
        prediction = model.predict(test_img_expanded)
        predicted_label_index = np.argmax(prediction)
        actual_label_index = np.argmax(y_test[index])
        predicted_label = categories[predicted_label_index]
        actual_label = categories[actual_label_index]

        if predicted_label == actual_label:
            right_answers += 1
        else:
            wrong_answers += 1

        print(f"Predicted: {predicted_label} \tActual: {actual_label}")

    print(f"Right answers: {right_answers}")
    print(f"Wrong answers: {wrong_answers}")


# Main execution flow
def main():
    # Load the test data
    X_test, y_test_raw = load_test_data(TEST_DIR, IMG_SIZE, VALID_EXTENSIONS)
    print(f"Loaded {len(X_test)} test images.")

    # Preprocess the test data
    X_test, y_test, label_encoder = preprocess_data(X_test, y_test_raw, CATEGORIES)
    print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")

    # Load the trained model
    model = load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    # Evaluate the model
    evaluate_model(model, X_test, y_test, CATEGORIES)


if __name__ == "__main__":
    main()
