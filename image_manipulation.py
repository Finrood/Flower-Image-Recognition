import os
import pickle
import random

import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.transform import resize

DATADIR = "Datasets/flowers/"
CATEGORIES = []

IMG_SIZE = (128, 128)

dirList = os.listdir(DATADIR)
training_data = []
imageData = []
imageLabel = []
validExtensionsList = ['jpg', 'tif', 'png', 'bmp']


def create_training_data():
    # A big number
    # The minimum number of images from all category
    min_images_in_categ = 1000000000
    min_images_categ = ''
    # Get the minimum amount of usable images for a category.
    # This will ensure we don't have more images in one category then another
    for d in dirList:
        current_image_count = 0
        path = os.path.join(DATADIR, d)
        for f in os.listdir(path):
            extension = f.split('.')[-1]
            if extension in validExtensionsList:
                img_path = os.path.join(DATADIR, d, f)
                try:
                    img = Image.open(img_path)
                except:
                    img = None
                if img is not None:
                    current_image_count += 1
        if current_image_count < min_images_in_categ:
            min_images_in_categ = current_image_count
            min_images_categ = d

    print("Minimum number of images per category : ", str(min_images_in_categ) + " for category " + min_images_categ)

    for d in dirList:
        print(f"processing... {d}")
        current_image_count = -10000000
        for f in os.listdir(DATADIR + d):
            extension = f.split('.')[-1]
            if extension in validExtensionsList:
                if current_image_count < min_images_in_categ:
                    img_path = os.path.join(DATADIR, d, f)
                    # First, we transform all images into RGB
                    try:
                        img = Image.open(img_path)
                    except:
                        img = None
                    if img is not None:
                        img = img.convert('RGB')
                        img.save(img_path)

                        # We load the image
                        data = imread(img_path)

                        resized_data = resize(data, IMG_SIZE)
                        training_data.append([resized_data, d])
                        current_image_count += 1
            else:
                pass
    print("size of training data : ", len(training_data))
    return training_data


def main():
    # The x axis is the pixel values
    X = []
    # The y axis is the labels
    y = []

    print("Turning the images in RGB")

    training_data = create_training_data()

    # We shuffle 3 time to be sure
    random.shuffle(training_data)
    random.shuffle(training_data)
    random.shuffle(training_data)

    for features, label in training_data:
        X.append(features)
        y.append(label)

    print("Reshaping the array of pixel values")
    # Reshape the array into a 4 dimensions array of shape [24946*[100*[100*[R,G,B]]]]
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 3)

    print("Saving the pickles")
    # We save our pixel array and label array in a pickle file
    # that can be reused later without having to do the processing again
    pickle_out = open("pickles/features.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("pickles/label.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


main()
