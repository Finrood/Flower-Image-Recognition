import os
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

DATADIR = Path("Datasets/flowers/")
IMG_SIZE = (128, 128)
VALID_EXTENSIONS = {'jpg', 'tif', 'png', 'bmp'}
PICKLE_DIR = Path("pickles")
PICKLE_DIR.mkdir(exist_ok=True)

def get_categories(datadir: Path) -> List[str]:
    return [d.name for d in datadir.iterdir() if d.is_dir()]

def get_image_paths(category_path: Path) -> List[Path]:
    return [p for p in category_path.iterdir() if p.suffix[1:] in VALID_EXTENSIONS]

def get_min_images_in_category(categories: List[str], datadir: Path) -> int:
    min_images = float('inf')
    for category in categories:
        image_count = len(get_image_paths(datadir / category))
        if image_count < min_images:
            min_images = image_count
    return min_images

def process_image(image_path: Path) -> Union[np.ndarray, None]:
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            data = imread(image_path)
            resized_data = resize(data, IMG_SIZE, anti_aliasing=True)
            return resized_data
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def create_training_data(datadir: Path, categories: List[str], min_images: int) -> List[Tuple[np.ndarray, str]]:
    training_data = []
    for category in categories:
        category_path = datadir / category
        image_paths = get_image_paths(category_path)[:min_images]
        for img_path in tqdm(image_paths, desc=f"Processing {category}"):
            processed_image = process_image(img_path)
            if processed_image is not None:
                training_data.append((processed_image, category))
    return training_data

def save_pickle(data: Union[np.ndarray, List], filename: Path):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def main():
    categories = get_categories(DATADIR)
    min_images = get_min_images_in_category(categories, DATADIR)

    print(f"Minimum number of images per category: {min_images}")

    training_data = create_training_data(DATADIR, categories, min_images)

    random.shuffle(training_data)

    X = np.array([features for features, _ in training_data])
    y = np.array([label for _, label in training_data])

    print("Saving the pickles")

    save_pickle(X, PICKLE_DIR / "features.pickle")
    save_pickle(y, PICKLE_DIR / "labels.pickle")

if __name__ == "__main__":
    main()
