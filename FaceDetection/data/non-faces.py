import numpy as np
import cPickle
import skimage.transform
import skimage.io
import os

print 'Loading dataset...'


data_file_name = 'data/cifar-10-batches-py/data_batch_1'
image_save_path = 'data/processed_faces/'


def extract_nonfaces(size):

    file_name = "{}_{}_nonfaces".format(*(size))

    new_dir_name = os.path.join(image_save_path, file_name)

    if not os.path.exists(new_dir_name):
        os.makedirs(new_dir_name)


    with open(data_file_name, 'rb') as f:
        data_dict = cPickle.load(f)

    data = data_dict['data'].reshape((-1, 32, 32, 3), order='F')
    sample_amount = data.shape[0]

    data_resized = np.zeros((sample_amount, size[0], size[1], 3), dtype='float64')


    for sample_number in xrange(sample_amount):
        current_sample = data[sample_number, :, :, :]
        data_resized[sample_number, :, :, :] = \
        skimage.transform.resize(current_sample, (size[0], size[1], 3))
        skimage.io.imsave(os.path.join(new_dir_name, str(sample_number) + '.jpg') , data_resized[sample_number, :, :, :])

if __name__ == '__main__':

    extract_nonfaces((13, 13))
    extract_nonfaces((25, 25))