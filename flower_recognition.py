import itertools
import os
import pickle
import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

dirList = os.listdir("datasets/flowers/")
imageData = []
imageLabel = []
validExtensionsList = ['jpg', 'tif', 'png', 'bmp']
tensorboard = None

config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

print("Reading pre-processed training data (pickles)")
pickle_features = pickle.load(open("pickles/features.pickle", "rb"))
pickle_label = pickle.load(open("pickles/label.pickle", "rb"))


# for d in dirList:
#     print(f"processing... {d}")
#     for f in os.listdir('datasets/flowers/'+d):
#         ext = f.split('.')[-1]
#         if ext in validExtensionsList:
#             img = Image.open(f"datasets/flowers/{d}/{f}").convert('RGB')
#             img.save(f'datasets/flowers/{d}/{f}')
#             data = imread(f"datasets/flowers/{d}/{f}")
#             #color.rgbgray(data)
#             resized_data = resize(data, (128,128))
#             imageData.append(resized_data)
#             imageLabel.append(d)
#         else:
#             pass
# print("DONE !")
# imageData = np.array(imageData)
# imageLabel = np.array(imageLabel)
# print(imageData)
# np.shape(imageData), np.shape(imageLabel)
#
# # idx = np.random.randint(len(imageData))
# # plt.imshow(imageData[idx])
# # plt.xlabel(imageLabel[idx])
# # plt.show()

def createCNNModel():
    dense_layers = [0, 1, 2, 3, 4, 5]
    layer_sizes = [16, 32, 64, 128, 256, 512, 1024]
    conv_layers = [0, 1, 2, 3, 4, 5]

    # for dense_layer in dense_layers:
    #     for layer_size in layer_sizes:
    #         for conv_layer in conv_layers:
    #             NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
    #             tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    #
    #             model = Sequential()

    NAME = f"Flowers_recognition-Default-{int(time.time())}.log"
    global tensorboard
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    model = Sequential()
    model.add(Conv2D(64, 3, input_shape=pickle_features.shape[1:], activation='relu'))
    model.add(Conv2D(64, 3, input_shape=pickle_features.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))

    model.add(Conv2D(128, 3, input_shape=pickle_features.shape[1:], activation='relu'))
    model.add(Conv2D(128, 3, input_shape=pickle_features.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))

    model.add(Conv2D(256, 3, input_shape=pickle_features.shape[1:], activation='relu'))
    model.add(Conv2D(256, 3, input_shape=pickle_features.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))

    model.add(Conv2D(512, 3, input_shape=pickle_features.shape[1:], activation='relu'))
    model.add(Conv2D(512, 3, input_shape=pickle_features.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.20))

    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.20))

    model.add(Dense(len(set(pickle_label)), activation='softmax'))
    return model


print("Creating the model")
model = createCNNModel()
model.summary()

epochs = 30
batch_size = 50
count = 0
count_limit = int(len(pickle_features) / 1)
while count != len(pickle_features):
    print("starting")
    # The x axis is the pixel values
    X = []
    # The y axis is the labels
    y = []
    for idx, val in enumerate(itertools.islice(pickle_features, count, count_limit)):
        X.append(pickle_features[count])
        y.append(pickle_label[count])
        count += 1

    X = np.array(X)
    y = np.array(y)

    # We use set so that we only have different labels. Not the sames
    num_classes = len(set(y))
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(num_classes=num_classes, y=y)
    input_shape = (X.shape[1], X.shape[2], X.shape[3])

    np.shape(X), np.shape(y)

    print(y)
    print(X.shape)
    print(y.shape)

    print("Setting up training and validation data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1)
    np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test), np.shape(X_val), np.shape(y_val)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    datagen.fit(X_train)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if count == len(pickle_features):
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs,
                         verbose=1, callbacks=[tensorboard])
    else:
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs,
                         verbose=1)
    count_limit += int(len(pickle_features) / 5)

model.save('models/Flower-test-10f.model')

model.evaluate(X_test, y_test)

plt.plot(range(len(hist.history['acc'])), hist.history['acc'])
plt.plot(range(len(hist.history['loss'])), hist.history['loss'])
plt.xlabel('epoch')
plt.show()

idxtest = np.random.randint(len(X_test))
testImg = X_test[idxtest]
plt.imshow(testImg, cmap='gray')
# cv2.resize(testImg, (128, 128))
testImg = testImg.reshape(1, 128, 128, 3)
print(np.array(imageLabel))
print(model.predict(testImg))
print(np.argmax(model.predict(testImg)))
plt.show()
pred = le.inverse_transform(np.argmax(model.predict(testImg)))
actual = le.inverse_transform(np.argmax(y_test[idxtest]))
print("Actual:", actual, " Predicted:", pred)
plt.show()

# img_data = []
# labels = []
#
# size = 128, 128
# def iter_images(images, directory, size, label):
#     try:
#         for i in range(len(images)):
#             img = cv2.imread(directory + images[i])
#             img = cv2.resize(img, size)
#             img_data.append(img)
#             labels.append(label)
#     except:
#         pass
#
#
# iter_images(listdir(daisy_path), daisy_path, size, 0)
# iter_images(listdir(daisy_path), dandelion_path, size, 1)
# iter_images(listdir(rose_path), rose_path, size, 2)
# iter_images(listdir(sunflower_path), sunflower_path, size, 3)
# iter_images(listdir(tulip_path), tulip_path, size, 4)
#
# data = np.asarray(img_data)
#
# #div by 255
# data = data.astype('float32') / 255.0
#
# labels = np.asarray(labels)
#
# # Split the data
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.30, shuffle=True)
#
# classes = 5
#
# y_train_binary = to_categorical(y_train, classes)
# y_test_binary = to_categorical(y_test, classes)
#
# NAME = "Flower_test"
# tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
#
#
# dense_layers = [1, 2, 3]
# layer_sizes = [16, 32, 64, 128]
# conv_layers = [0, 1, 2, 3]
#
# for dense_layer in dense_layers:
#     for layer_size in layer_sizes:
#         for conv_layer in conv_layers:
#             NAME = f"FlowersRecognition-{conv_layer}_conv-{layer_size}_nodes-{dense_layer}_dense-{int(time.time())}"
#             tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
#             model = Sequential()
#
#             #First layer
#             model.add(Conv2D(layer_size, (3, 3), input_shape=(128, 128, 3)))
#             model.add(Activation("relu"))
#             model.add(MaxPool2D(pool_size=(2, 2)))
#
#
#             for l in range(conv_layer-1):
#                 #Second layer
#                 model.add(Conv2D(layer_size, (3, 3), input_shape=(128, 128, 3)))
#                 model.add(Activation("relu"))
#                 model.add(MaxPool2D(pool_size=(2, 2)))
#
#             model.add(Flatten())
#
#             for l in range(dense_layer):
#                 model.add(Dense(layer_size))
#                 model.add(Activation("relu"))
#
#             #Output layer
#             model.add(Dense(5))
#             model.add(Activation("softmax"))
#
#             model.compile(loss="binary_crossentropy", optimizer="adadelta", metrics=["accuracy"])
#
#             batch_size = 128
#             epochs = 30
#             model.fit(x_train, y_train_binary, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test_binary), callbacks=[tensorboard])
#
#             model.save(f"models/{NAME}.model")
