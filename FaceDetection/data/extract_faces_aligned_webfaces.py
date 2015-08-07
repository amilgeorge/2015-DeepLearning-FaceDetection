import os
import sys
from PIL import Image

sys.path.append("data")

from crop_face import CropFace

CUR_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CUR_FILE_DIR, os.pardir))

images_path = os.path.join(BASE_DIR, 'data/Caltech_WebFaces/')
image_save_path = os.path.join(BASE_DIR, 'data/processed_images/')
face_description_file_path = os.path.join(BASE_DIR, 'data/WebFaces_GroundThruth.txt')


def extract_faces(output_size, number_of_faces_to_fetch, offset=(0.3, 0.2)):

    file_name = "{}_{}_faces_web".format(*(output_size))
    new_dir_name = os.path.join(image_save_path, file_name)

    if not os.path.exists(new_dir_name):
        os.makedirs(new_dir_name)

    file_handler = open(face_description_file_path)
    lines = file_handler.readlines()
    file_handler.close()

    faces_data = map(lambda x: x.strip().split(), lines)

    for face_number, face_description in enumerate(faces_data):

        if face_number >= number_of_faces_to_fetch:
            break

        face_image_name = face_description[0]
        face_image_path = os.path.join(images_path, face_image_name)
        face_image_save_path = os.path.join(new_dir_name, face_image_name)
        img = Image.open(face_image_path)

        face_eyes_coordinates = map(lambda x: float(x), face_description[1:5])

        eye_left = face_eyes_coordinates[:2]
        eye_right = face_eyes_coordinates[2:]

        cropped_face_img = CropFace(img, eye_left=eye_left, eye_right=eye_right,
                                    offset_pct=offset, dest_sz=output_size)

        cropped_face_img.save(face_image_save_path)

        print "Face number {} is processed.".format(face_number)

if __name__ == '__main__':
    extract_faces((48, 48), 10000)
    #extract_faces((25, 25))

