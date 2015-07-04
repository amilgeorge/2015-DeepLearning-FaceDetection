import skimage.io as io
from os import walk
import re
import math
import os

from skimage.transform import resize

FDDB_path = './data/FDDB-folds/'
image_path = './data/originalPics/'
image_save_path = './data/processed_faces/'


def extract_faces(output_size):

    file_name = "{}_{}_faces".format(*(output_size))

    new_dir_name = os.path.join(image_save_path, file_name)

    if not os.path.exists(new_dir_name):
        os.makedirs(new_dir_name)

    f = []
    for (dirpath, dirnames, filenames) in walk(FDDB_path):
        f.extend(filenames)
        break

    regex = re.compile(".*(ellipseList).*")
    f = [m.group(0) for l in f for m in [regex.search(l)] if m]

    face_save_count = 0

    for data_file in f:

        file = open(FDDB_path + data_file, 'rb')
        n = -1


        lines = file.readlines()
        lines = map(lambda x: x.strip(), lines)
        n_lines = len(lines)
        i = 0


        while(i < n_lines):

            path = image_path + lines[i] + '.jpg'
            print path
            image = io.imread(path)

            image_max_row = image.shape[0] - 1
            image_max_col = image.shape[1] - 1
            i += 1

            n_faces = lines[i]
            i+=1

            for j in xrange(int(n_faces)):

                param =  lines[i].split()
                param = map(lambda x: float(x), param)


                major_axis_radius = param[0]
                minor_axis_radius = param[1]
                angle = math.radians(param[2])
                center_x= param[3]
                center_y= param[4]
                detection_score= param[5]

                row = center_y - minor_axis_radius
                col = center_x - minor_axis_radius

                width = minor_axis_radius * 2
                height = minor_axis_radius * 2

                # If the bounding box doesn't fit, we just discard the image.
                if (row < 0 or col < 0 or row > image_max_row or col > image_max_col):
                    i += 1
                    continue

                im = image[row:row+height, col:col+width]

                im = resize(im, output_size)

                save_file_string = os.path.join(new_dir_name, "{0:06d}".format(face_save_count) + '.jpg')

                if len(image.shape) < 3:
                    i += 1
                    continue

                if image.shape[2] == 1:
                    i += 1
                    continue

                io.imsave(save_file_string, im)

                face_save_count += 1
                i += 1

if __name__ == '__main__':

    extract_faces((13, 13))
    extract_faces((25, 25))


