import os
import pickle
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Dropout, MaxPooling2D, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras_tuner import HyperModel, RandomSearch

# Set directories
PICKLE_DIR = Path("pickles")
LOG_DIR = Path("logs")

# Load preprocessed data
print("Reading pre-processed training data (pickles)")
X = pickle.load(open(PICKLE_DIR / "features.pickle", "rb"))
y = pickle.load(open(PICKLE_DIR / "labels.pickle", "rb"))

# Data normalization
X = X / 255.0

# Label encoding
label_encoder = LabelEncoder()
y_indices = label_encoder.fit_transform(y)
y_one_hot = to_categorical(y_indices, num_classes=len(np.unique(y)))

# Train-validation-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.20, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Define the HyperModel
class FlowerHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=X.shape[1:]))
        model.add(Conv2D(hp.Int('conv_1_filters', min_value=32, max_value=256, step=32),
                         kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))

        for i in range(hp.Int('num_conv_layers', 1, 4)):
            model.add(Conv2D(hp.Int(f'conv_{i+2}_filters', min_value=32, max_value=256, step=32),
                             kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(BatchNormalization())
            model.add(Dropout(hp.Float(f'dropout_{i+2}', min_value=0.2, max_value=0.5, step=0.1)))

        model.add(Flatten())

        for i in range(hp.Int('num_dense_layers', 1, 3)):
            model.add(Dense(hp.Int(f'dense_{i+1}_units', min_value=128, max_value=1024, step=128), activation='relu'))
            model.add(Dropout(hp.Float(f'dropout_dense_{i+1}', min_value=0.2, max_value=0.5, step=0.1)))

        model.add(Dense(len(np.unique(y)), activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

# Hyperparameter tuning
hypermodel = FlowerHyperModel()

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=2,
    directory=LOG_DIR,
    project_name='flower_classification'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
tensorboard = TensorBoard(log_dir=LOG_DIR / f"fit_{int(time.time())}")
model_checkpoint = ModelCheckpoint(filepath='best_model.keras', save_best_only=True, monitor='val_loss')

tuner.search(datagen.flow(X_train, y_train, batch_size=32),
             validation_data=(X_val, y_val),
             epochs=30,
             callbacks=[early_stopping, tensorboard, model_checkpoint])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Evaluate the model
best_model.evaluate(X_test, y_test)

# Save the best model
best_model.save('best_flower_model.keras')

# Plot training history
history = best_model.history.history
plt.plot(history['accuracy'], label='accuracy')
plt.plot(history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Make a prediction
idx = np.random.randint(len(X_test))
test_img = X_test[idx]
plt.imshow(test_img)
plt.show()
test_img = np.expand_dims(test_img, axis=0)
pred = np.argmax(best_model.predict(test_img), axis=1)
actual = np.argmax(y_test[idx])
print(f"Actual: {label_encoder.inverse_transform([actual])[0]}, Predicted: {label_encoder.inverse_transform(pred)[0]}")
