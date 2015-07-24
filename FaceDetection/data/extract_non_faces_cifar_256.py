__author__ = 'Tanuj'

import os
import os.path
import skimage.io as io
from fnmatch import fnmatch
import numpy as np

from skimage.transform import resize
from skimage.util.shape import view_as_windows

CUR_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CUR_FILE_DIR, os.pardir))

image_path = os.path.join(BASE_DIR, 'data/256_ObjectCategories/')
image_save_path = os.path.join(BASE_DIR, 'data/processed_images/non-faces/')

face_dir_pattern = "253*.jpg"


def extract_non_faces(output_size):
    file_name = "{}_{}_non_faces".format(*(output_size))
    new_dir_name = os.path.join(image_save_path, file_name)

    i = 0

    if not os.path.exists(new_dir_name):
        os.makedirs(new_dir_name)

    file_paths = []

    for path, subdirs, files in os.walk(image_path):
        for name in files:
            if not fnmatch(name, face_dir_pattern):
                file_paths.append(os.path.join(path, name))

    window_shape = (32, 32)

    for file_path in file_paths:
        image = io.imread(file_path)

        if len(image.shape) < 3:
            continue

        if image.shape[2] == 1:
            continue

        windows = view_as_windows(image, window_shape)

        for window in windows:
            save_file_string = os.path.join(new_dir_name, "{0:06d}".format(i) + ".jpg")

            im = resize(window, output_size)
            io.imsave(save_file_string, im)

            i += 1

if __name__ == '__main__':
    extract_non_faces((13, 13))