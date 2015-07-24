__author__ = 'Tanuj'

import urllib
import os.path

CUR_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CUR_FILE_DIR, os.pardir))

image_save_path = os.path.join(BASE_DIR, 'data/haar/')

BASE_URL = "http://tutorial-haartraining.googlecode.com/svn/trunk/data/negatives/"

UMD_FILE_NAME_STR = "UMD_"
NO_IMAGES_UMD = 106
NEG_FILE_NAME_STR = "neg-"
NO_IMAGES_NEG = 4875


def fetch_images():
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    file_names = []
    # file_save_paths = []

    for i in range(1, NO_IMAGES_UMD + 1):
        file_name = UMD_FILE_NAME_STR + "{0:03d}".format(i) + ".jpg"
        # file_save_path = os.path.join(image_save_path, file_name)

        # urllib.urlretrieve(BASE_URL + file_name, file_save_path)

        file_names.append(file_name)
        # file_save_paths.append(file_save_path)

    for i in range(2, NO_IMAGES_NEG + 1):
        file_name = NEG_FILE_NAME_STR + "{0:04d}".format(i) + ".jpg"
        # file_save_path = os.path.join(image_save_path, file_name)

        # urllib.urlretrieve(BASE_URL + file_name, file_save_path)

        file_names.append(file_name)
        # file_save_paths.append(file_save_path)

    i = 0
    for file_name in file_names:
        file_save_path = os.path.join(image_save_path, file_name)
        urllib.urlretrieve(BASE_URL + file_name, file_save_path)

if __name__ == '__main__':
    fetch_images()