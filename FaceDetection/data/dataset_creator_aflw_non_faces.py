__author__ = 'Tanuj'

import sqlite3
import os.path
import skimage.io as io
import numpy as np

from skimage.filters import gaussian_filter
from skimage.util import random_noise
from skimage import img_as_ubyte

CUR_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CUR_FILE_DIR, os.pardir))

db_path = os.path.join(BASE_DIR, r"data/aflw/data/aflw.sqlite")
image_path = os.path.join(BASE_DIR, 'data/aflw/data/flickr/')
dataset_save_path = os.path.join(BASE_DIR, 'data/processed_images/aflw/non_faces')
image_save_path = os.path.join(BASE_DIR, 'data/processed_images/')


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
    modified = False

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

                mean = np.mean(im)

                im = np.empty(shape=im.shape)
                im.fill(mean)
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

            # if face_save_count == amount_of_faces_to_extract:
            #     break

        if modified:
            save_file_string = os.path.join(dataset_save_path, "{0:06d}".format(image_save_count) + ".jpg")
            io.imsave(save_file_string, image)
            # io.imsave(os.path.join(dataset_save_path, path), image)

            if image_save_count % 10 == 0:
                print "Processed", image_save_count, "/", no_images,  "images"

            image_save_count += 1

    c.close()

if __name__ == '__main__':
    create_non_faces_dataset_aflw()