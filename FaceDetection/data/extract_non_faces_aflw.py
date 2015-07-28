__author__ = 'Tanuj'

import sqlite3
import os.path
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from os import walk

from skimage.filters import gaussian_filter
from skimage.util import random_noise
from skimage import img_as_ubyte
from skimage.util.shape import view_as_windows
from skimage.transform import resize
from skimage.util.shape import view_as_blocks

CUR_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CUR_FILE_DIR, os.pardir))

db_path = os.path.join(BASE_DIR, r"data/aflw/data/aflw.sqlite")
image_path = os.path.join(BASE_DIR, 'data/aflw/data/flickr/')
dataset_save_path = os.path.join(BASE_DIR, 'data/processed_images/aflw/non_faces')
image_save_path = os.path.join(BASE_DIR, 'data/processed_images/')

output_fixed_size = (24, 24)

def create_non_faces_dataset_aflw():
    # file_name = "{}_{}_nonfaces_aflw".format(*(output_size))
    # new_dir_name = os.path.join(image_save_path, file_name)

    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)

    with sqlite3.connect(db_path) as db:
        c = db.cursor()

        query = "select count(file_id) from FaceImages;"

        no_images = c.execute(query).fetchone()[0]

        query = "select Faces.file_id, count(Faces.face_id), filepath from Faces inner join FaceImages on " \
                "Faces.file_id=FaceImages.file_id group by Faces.file_id;"

        res = c.execute(query)

    face_images = res.fetchall()

    image_save_count = 0
    face_save_count = 0

    for face_image in face_images:
        file_id = face_image[0]
        no_faces = face_image[1]
        image_file_path = face_image[2]

        path = image_path + image_file_path

        image = io.imread(path)

        if len(image.shape) < 3:
            continue

        if image.shape[2] == 1:
            continue

        query = "select file_id, Faces.face_id, x, y, w, h from Faces inner join FaceRect on " \
                "Faces.face_id=FaceRect.face_id where file_id='" + file_id + "';"

        res = c.execute(query)

        modified = False

        for i in xrange(no_faces):
            face = res.fetchone()

            # print face

            try:
                col = face[2]
                row = face[3]
                width = face[4]
                height = face[5]

                image_max_row = image.shape[0]
                image_max_col = image.shape[1]

                if row < 0 or col < 0 or row > image_max_row or col > image_max_col:
                    continue

                im = image[row: row + height, col: col + width]
                im = gaussian_filter(im, sigma=50)
                im = random_noise(im)
                im = img_as_ubyte(im)
                # show_images(im)

                image[row: row + height, col: col + width] = im[:]
                modified = True
                # show_images(image)

            except:
                print("Exception")
                continue

            face_save_count += 1
            #
            # if face_save_count == amount_of_faces_to_extract:
            #     break

        if modified:
            save_file_string = os.path.join(dataset_save_path, "{0:06d}".format(image_save_count) + ".jpg")
            io.imsave(save_file_string, image)

            if image_save_count % 10 == 0:
                print "Processed", image_save_count, "/", no_images,  "images"

            image_save_count += 1

    c.close()


def extract_non_faces(output_size, no_negative_samples_required):
    file_name = "{}_{}_non_faces_aflw".format(*(output_size))

    new_dir_name = os.path.join(image_save_path, file_name)

    if not os.path.exists(new_dir_name):
        os.makedirs(new_dir_name)

    image_files = []
    for (dirpath, dirnames, filenames) in walk(dataset_save_path):
        image_files.extend(filenames)
        break

    output_size = (output_size[0], output_size[1], 3)
    image_save_count = 0
    no_available_images = len(image_files)
    no_samples_per_image = 1.0 * no_negative_samples_required / no_available_images

    if no_samples_per_image < 1:
        no_samples_per_image = 2

    for image_file in image_files:
        image_file_path = os.path.join(dataset_save_path, image_file)
        image = io.imread(image_file_path)

        if len(image.shape) < 3:
            continue

        if image.shape[2] == 1:
            continue

        step_size = int(image.shape[0] / 10)

        images = view_as_windows(image, output_size, step_size)
        # images = view_as_blocks(image, output_size)

        random_windows = []
        selected_windows_dict = {}

        max_col_no = images.shape[0]
        max_row_no = images.shape[1]

        for i in xrange(no_samples_per_image):
            is_unique_key = False

            while not is_unique_key:
                col = np.random.randint(0, max_col_no)
                row = np.random.randint(0, max_row_no)

                key = str(col) + '_' + str(row)

                if not selected_windows_dict.has_key(key):
                    is_unique_key = True
                    selected_windows_dict[key] = True

            random_windows.append(images[col, row, 0])

        for im in random_windows:
            save_file_string = os.path.join(new_dir_name, "{0:06d}".format(image_save_count) + '.jpg')
            im = resize(im, output_fixed_size)
            io.imsave(save_file_string, im)
            image_save_count += 1

        if image_save_count % 50 == 0:
            print "Processed", image_save_count, "/", no_negative_samples_required,  "samples"

        if image_save_count == no_negative_samples_required:
            break


def show_images(images, titles=None):
    """Display a list of images"""
    if images is not list:
        images = [images]

    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n) # Make subplot
        if image.ndim == 2: # Is image grayscale?
            plt.gray() # Only place in this blog you can't replace 'gray' with 'grey'
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

if __name__ == '__main__':
    extract_non_faces((48, 48), 1000)
