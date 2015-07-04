

import numpy as np
import cPickle
import skimage.transform
import skimage.io

print 'Loading dataset...'


data_file_name = 'data_batch_1'
with open(data_file_name, 'rb') as f:
    data_dict = cPickle.load(f)

data = data_dict['data'].reshape((-1, 32, 32, 3), order='F')
sample_amount = data.shape[0]

data_resized = np.zeros((sample_amount, 13, 13,3), dtype='float64')    #np.zeros((sample_amount, 32, 32), dtype='float64') 
   
savepath = './data/non-faces/'

for sample_number in xrange(sample_amount):
    current_sample = data[sample_number, :, :, :]
    # Resize and convert to gray by averaging R, G, B.
    data_resized[sample_number, :, :, :] = \
            skimage.transform.resize(current_sample, (13, 13, 3)) # skimage.transform.resize(current_sample, (32, 32)).mean(axis=2)
    skimage.io.imsave(savepath + str(sample_number) + '.jpg' , data_resized[sample_number, :, :, :] )        
