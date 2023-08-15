import os
import numpy as np
from imageio import imread
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

dirList = os.listdir("datasets/test_flowers/")
imageData = []
imageLabel = []
validExtensionsList = ['jpg', 'tif', 'png', 'bmp']
tensorboard = None
CATEGORIES = ['daisy', 'dandelion', 'Gerbera', 'Iris', 'jonquille', 'Lilac', 'Orchid', 'rose', 'sunflower', 'tulip']

for d in dirList:
    print(f"processing... {d}")
    for f in os.listdir('datasets/test_flowers/'+d):
        ext = f.split('.')[-1]
        if ext in validExtensionsList:
            data = imread(f"datasets/test_flowers/{d}/{f}")
            #color.rgbgray(data)
            resized_data = resize(data, (128,128))
            imageData.append(resized_data)
            imageLabel.append(d)
        else:
            pass
print("DONE !")

np.shape(imageData), np.shape(imageLabel)

X = np.array(imageData)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2],3)
y = np.array(imageLabel)
num_classes = len(set(y))
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(num_classes=num_classes, y=y)
input_shape = (X.shape[1], X.shape[2], X.shape[3])

np.shape(X), np.shape(y)

model = tf.keras.models.load_model('models/Flower-test-10f.model')

# model.evaluate(X, y)
right_answer = 0
wrong_answer = 0
index = 0
for testImg in X:
    # plt.imshow(testImg, cmap='gray')
    testImg = testImg.reshape(1, 128, 128, 3)
    labelIndex = np.argmax(model.predict(testImg))
    deduction = CATEGORIES[labelIndex]
    reponse = CATEGORIES[np.argmax(y[index])]
    print(f"deduction = {deduction} \t \t \tréponse = {reponse}")
    if deduction == reponse:
        right_answer += 1
    else:
        wrong_answer += 1
    index += 1
    # plt.show()
print(f"Right answers : {right_answer}")
print(f"Wrong answers : {wrong_answer}")



# CATEGORIES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
#
# test_sample = ["dandel2.jpg", "tulip.jpg", "dandel.jpg", "rose.jpg", "proxy.duckduckgo.com.jpg"]
# imageData = []
#
# model = tf.keras.models.load_model('Flower-test.model')
#
# IMG_SIZE = 128
#
# for image in test_sample:
#     data = imread(image)
#     resized_data = resize(data, (IMG_SIZE, IMG_SIZE))
#     imageData.append(resized_data)
#
# np.shape(imageData)
# X = np.array(imageData)
# X = X.reshape(-1, IMG_SIZE,  IMG_SIZE, 3)
# np.shape(X)
#
# for testImg in X:
#     plt.imshow(testImg, cmap='gray')
#     testImg = testImg.reshape(1, IMG_SIZE, IMG_SIZE, 3)
#     print(model.predict(testImg))
#     labelIndex = np.argmax(model.predict(testImg))
#     deduction = CATEGORIES[labelIndex]
#     print(deduction)
#     plt.show()
