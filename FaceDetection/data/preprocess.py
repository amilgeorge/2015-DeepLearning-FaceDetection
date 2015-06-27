import skimage.io as io
from os import walk
import re
import math

from skimage.transform import resize

FDDB_path = './data/FDDB-folds/'
image_path = './data/raw_images/'
image_save_path = './data/raw_images/train_faces/'

def extract_faces(output_folder, output_size):

    f = []
    for (dirpath, dirnames, filenames) in walk(FDDB_path):
        f.extend(filenames)
        break

    regex = re.compile(".*(ellipseList).*")
    f = [m.group(0) for l in f for m in [regex.search(l)] if m]

    file = open(FDDB_path + f[0], 'rb')
    n = -1


    lines = file.readlines()
    lines = map(lambda x: x.strip(), lines)
    n_lines = len(lines)
    i = 0
    face_save_count = 0

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

            save_file_string = output_folder + "{0:06d}".format(face_save_count) + '.jpg'

            io.imsave(save_file_string, im)

            face_save_count += 1
            i += 1

if __name__ == '__main__':

    extract_faces(image_save_path + '13/', (13, 13))
    extract_faces(image_save_path + '25/', (25, 25))


