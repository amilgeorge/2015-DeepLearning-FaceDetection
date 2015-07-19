__author__ = 'Tanuj'

import sqlite3
import os.path
import skimage.io as io

from skimage.transform import resize

CUR_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CUR_FILE_DIR, os.pardir))

db_path = os.path.join(BASE_DIR, r"data/aflw/data/aflw.sqlite")
image_path = os.path.join(BASE_DIR, 'data/aflw/data/flickr/')
image_save_path = os.path.join(BASE_DIR, 'data/processed_images/')


def extract_faces(output_size):
    file_name = "{}_{}_faces".format(*(output_size))
    new_dir_name = os.path.join(image_save_path, file_name)

    if not os.path.exists(new_dir_name):
        os.makedirs(new_dir_name)

    with sqlite3.connect(db_path) as db:
        c = db.cursor()

        query = "select count(face_id) from FaceRect;"

        no_rows = c.execute(query).fetchone()[0]

        query = "SELECT FaceRect.face_id , FaceRect.x, FaceRect.y, FaceRect.w, FaceRect.h, FaceImages.file_id, " \
                "FaceImages.filepath, FaceImages.width, FaceImages.height from FaceRect  inner join Faces on " \
                "FaceRect.face_id = Faces.face_id inner join FaceImages on Faces.file_id = FaceImages.file_id"
        #
        res = c.execute(query)

    faces = []

    face_save_count = 0

    for i in range(no_rows):
        face = res.fetchone()

        col = face[1]
        row = face[2]
        width = face[3]
        height = face[4]

        image_file_path = face[6]
        path = image_path + image_file_path

        image = io.imread(path)

        if len(image.shape) < 3:
            continue

        if image.shape[2] == 1:
            continue

        image_max_row = image.shape[0] - 1
        image_max_col = image.shape[1] - 1

        if row < 0 or col < 0 or row > image_max_row or col > image_max_col:
            continue

        im = image[row: row + height, col: col + width]
        im = resize(im, output_size)

        save_file_string = os.path.join(new_dir_name, "{0:06d}".format(face_save_count) + ".jpg")

        io.imsave(save_file_string, im)
        face_save_count += 1

if __name__ == '__main__':
    extract_faces((13, 13))
    extract_faces((25, 25))