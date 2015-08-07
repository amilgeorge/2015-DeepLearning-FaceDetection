"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
import cPickle as pickle
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from skimage.io.collection import ImageCollection, concatenate_images
#from matplotlib.image import imread
from skimage.transform._warps import resize
from skimage.io._io import imread_collection

from matplotlib import pyplot as plt
from matplotlib import patches
from relu import relu
import matplotlib.cm as cm
import skimage.io as io

from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

from PIL import Image
from skimage.util.dtype import img_as_ubyte


SAVE_STATE_TO_FILE = "05_Aug_dataset2.save"
LOAD_STATE_FROM_FILE = "05_Aug_dataset2.save"

# settings for LBP
lbp_radius = 3
#lbp_n_points = 8 * radius
lbp_method = 'uniform'

NUM_CHANNELS = 1;

def save_list_of_weights(filename, weights_list):
    with open(filename, 'wb') as output:
        pickle.dump(weights_list, output, pickle.HIGHEST_PROTOCOL)

class twelve_net():
    
    def __init__(self, input, batch_size, activation, state=None):
        
        #layer0_input = x.reshape((batch_size, 3, 13, 13))
        rng = np.random.RandomState(23455)

        img_size = 13
        img_channels = NUM_CHANNELS

        conv_filter_size = 3
        conv_filter_stride = 1 # hard coded
        conv_filter_depth = 16

        ## Not used becusee it is hardcoded  inside le-net
        pool_filter_size = 3
        pool_filter_stride = 2

        conv_pool_output_size = 5 ## 10

        fullyconnected_output_size = 16

        self.input = input

        if state is None:
            conv_pool_layer_state = None
            fully_connected_layer_state = None
            log_regression_layer_state = None
        else:
            conv_pool_layer_state = state[0:2]
            fully_connected_layer_state = state[2:4]
            log_regression_layer_state = state[4:6]

        self.conv_pool_layer = LeNetConvPoolLayer(
            rng,
            input=input,
            image_shape=(batch_size, img_channels, img_size, img_size),
            filter_shape=(conv_filter_depth, img_channels, conv_filter_size, conv_filter_size),
            poolsize=(3, 3),
            activation=activation,
            state = conv_pool_layer_state
        )


        self.fullyconnected_layer = HiddenLayer(
        rng,
        input=self.conv_pool_layer.output.flatten(2),
        n_in=conv_filter_depth * conv_pool_output_size * conv_pool_output_size,
        n_out=fullyconnected_output_size,
        activation=activation,
        state = fully_connected_layer_state
        )

        self.log_regression_layer = LogisticRegression(input=self.fullyconnected_layer.output,
                                            n_in=fullyconnected_output_size,
                                            n_out=2, state = log_regression_layer_state)
        
        self.L1 = (
            abs(self.conv_pool_layer.W).sum()
            + abs(self.fullyconnected_layer.W).sum()
            + abs(self.log_regression_layer.W).sum()
        )

        self.params = self.conv_pool_layer.params + self.fullyconnected_layer.params + self.log_regression_layer.params

    def save(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self.params, output, pickle.HIGHEST_PROTOCOL)

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(3, 3), activation = None, state = None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))

        # Variance computation based on input for relu
        variance =.01# np.sqrt(2.0/fan_in)

        print variance

        if state is None:
        # initialize weights with random weights

            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W = theano.shared(
                np.asarray(
                    #rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    rng.normal(loc=0, scale=variance, size=filter_shape),
                    dtype=theano.config.floatX  # @UndefinedVariable
                ),
                borrow=True
            )

            # the bias is a 1D tensor -- one bias per output feature map
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)  # @UndefinedVariable
            b = theano.shared(value=b_values, borrow=True)

        else:
            W = theano.shared(
                np.asarray(
                    state[0],
                    dtype=theano.config.floatX, # @UndefinedVariable
                ),
                borrow=True
            )
            b = theano.shared(value=state[1], borrow=True)

        self.W = W
        self.b = b

        conv_layer_stride = 1;
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            subsample=(conv_layer_stride,conv_layer_stride)
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True,
            st = (2,2)
        )

        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        # Non linearity relu is suggested in paper
        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

def evaluate_12net(learning_rate=0.001, n_epochs=1000,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500, l1_lambda = 0.5):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = np.random.RandomState(23455)


    train_set_x, train_set_y = get_train_data()
    valid_set_x, valid_set_y = get_valid_data()
    test_set_x, test_set_y = get_train_data()

    # compute number of minibatches for training, validation and testing
    num_train_samples = train_set_x.get_value(borrow=True).shape[0]
    num_valid_samples = valid_set_x.get_value(borrow=True).shape[0]
    num_test_samples = test_set_x.get_value(borrow=True).shape[0]

    n_train_batches = num_train_samples/batch_size
    n_valid_batches = num_valid_samples/batch_size
    n_test_batches = num_test_samples/batch_size


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, NUM_CHANNELS, 13, 13))
    #layer0_input = x

    net = twelve_net(layer0_input, batch_size, relu)

    cost = net.log_regression_layer.negative_log_likelihood(y) #+ l1_lambda * net.L1
    errors = net.log_regression_layer.errors(y)
    
    

        # create a list of all model parameters to be fit by gradient descent
    params = net.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]


    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_err_model = theano.function(
        [index],
        errors,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    test_model = theano.function(
        [index],
        errors,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        errors,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


        ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    best_params = map(lambda x: x.eval(), net.params)

    # compute zero-one loss on validation set
    validation_losses = [validate_model(i) for i
                                in xrange(n_valid_batches)]
    this_validation_loss = np.mean(validation_losses)
    
    train_losses = [train_err_model(i) for i
                                in xrange(n_train_batches)]
    this_train_loss = np.mean(train_losses)
    print('Pre training  validation error %f %%' %
            (this_validation_loss * 100.))

    plt.ion()
    fig,axarr = plt.subplots(4,4)
    
    prev_W = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            
#             filter_W_tensor = net.conv_pool_layer.params[0]
#             filter_W = filter_W_tensor.eval()
#             visualize(fig, axarr, filter_W)
#             
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                train_losses = [train_err_model(i) for i
                                     in xrange(n_train_batches)]
                this_train_loss = np.mean(train_losses)
                print('epoch %i, minibatch %i/%i, train error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_train_loss * 100.))
                # if we got the best validation score until now
                filter_W_tensor = net.conv_pool_layer.params[0]
                filter_W = filter_W_tensor.eval()
                     
                diff = filter_W - prev_W
                prev_W = filter_W
                #visualize(fig, axarr, filter_W)
                
                if this_train_loss == 0:
                    done_looping = True
                
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    best_params = map(lambda x: x.eval(), net.params)

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    save_list_of_weights(SAVE_STATE_TO_FILE, best_params)

    #net.save(SAVE_STATE_TO_FILE)

# def convert_image_stack_to_lbp(img_arr):
# 
#     # settings for LBP
#     lbp_radius = 3
#     #lbp_n_points = 8 * radius
#     lbp_method = 'uniform'
# 
#     lbp_stack = np.zeros((img_arr.shape[0], img_arr.shape[1], img_arr.shape[2], 1))
#     img_amount = img_arr.shape[0]
# 
#     for img_count in xrange(img_amount):
# 
#         lbp_stack[img_count, :, :, 0] = local_binary_pattern(rgb2gray(img_arr[img_count, :, :, :]),
#                                                              lbp_n_points, lbp_radius, lbp_method)
# 
#     #print not lbp_stack.any()
# 
#     return lbp_stack


def prepare_data(faces_collection, bkgs_collection):

    arr_faces = concatenate_images(faces_collection)
    #arr_faces = convert_image_stack_to_lbp(arr_faces)
    arr_faces=arr_faces[:,:,:,np.newaxis]
    arr_faces = np.rollaxis(arr_faces, 3, 1)
    arr_faces = arr_faces
    num_face_imgs = arr_faces.shape[0]
    arr_faces = arr_faces.reshape((arr_faces.shape[0], -1)); # Need to check this ---compare with flatten used during training

    out_faces = np.ones(arr_faces.shape[0])

    arr_bkgs = concatenate_images(bkgs_collection)
    #arr_bkgs = convert_image_stack_to_lbp(arr_bkgs)
    arr_bkgs=arr_bkgs[:,:,:,np.newaxis]
    arr_bkgs = np.rollaxis(arr_bkgs, 3, 1)
    arr_bkgs = arr_bkgs.reshape((arr_bkgs.shape[0], -1));
    arr_bkgs = arr_bkgs # Reduce the size of bkg images
    out_bkgs = np.zeros(arr_bkgs.shape[0])

    test_set = np.concatenate((arr_faces, arr_bkgs))
    labels = np.concatenate((out_faces, out_bkgs))

    arr_indexes = np.random.permutation(test_set.shape[0])

    shuffled_test_set = test_set[arr_indexes]
    shuffled_labels = labels[arr_indexes].flatten()

#     borrow = True
#     shared_x = theano.shared(np.asarray(shuffled_test_set, dtype=theano.config.floatX),  # @UndefinedVariable
#                              borrow=borrow)
# 
#     shared_y = theano.shared(np.asarray(shuffled_labels, dtype=theano.config.floatX),  # @UndefinedVariable
#                              borrow=borrow)

    return shuffled_test_set,shuffled_labels

def get_valid_data():
    #/Users/amilgeorge/Documents/StudySS2015/DeepLearning/Training\ Data/data/raw_images/train_faces/
    img_faces = imread_collection("data/train_set/dataset/13/validation/faces/*.png")
    #
    img_bkgs = imread_collection("data/train_set/dataset/13/validation/nonfaces/*.jpg")

    #img_faces = imread_collection("data/processed_images/13_train_set_aflw/validation/faces/*.jpg")

    #img_bkgs = imread_collection("data/processed_images/13_train_set_aflw/validation/nonfaces/*.jpg")

    x, y = prepare_data(img_faces, img_bkgs)
    
    borrow = True
    shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX),  # @UndefinedVariable
                             borrow=borrow)

    shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX),  # @UndefinedVariable
                             borrow=borrow)

    return shared_x,T.cast(shared_y, 'int32')

def get_train_data():
    #/Users/amilgeorge/Documents/StudySS2015/DeepLearning/Training\ Data/data/raw_images/train_faces/
    img_faces = imread_collection("data/train_set/dataset/13/train/faces/*.png")
    #
    img_bkgs = imread_collection("data/train_set/dataset/13/train/nonfaces/*.jpg")

    #img_faces = imread_collection("data/processed_images/13_train_set_aflw/train/faces/*.jpg")

    #img_bkgs = imread_collection("data/processed_images/13_train_set_aflw/train/nonfaces/*.jpg")
    
    
    x, y = prepare_data(img_faces, img_bkgs)
    #x=x[:1000]
    #y=y[:1000]
    
    borrow = True
    shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX),  # @UndefinedVariable
                             borrow=borrow)

    shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX),  # @UndefinedVariable
                             borrow=borrow)
    return shared_x,T.cast(shared_y, 'int32')




# def get_test_data():
# 
#     img1 = imread('data/test/img_1.jpg')
#     testimg1 = resize(img1, (13, 13)).flatten()
# 
#     img2 = imread('data/test/img_47.jpg')
#     testimg2 = resize(img2, (13, 13)).flatten()
#     t = testimg1[np.newaxis, ...]
# 
#     list_imgs = np.concatenate([testimg1[np.newaxis, ...], testimg2[np.newaxis, ...]])
# 
#     #list_imgs = np.rollaxis(list_imgs,3,1)
# 
#     borrow = True
#     shared_x = theano.shared(np.asarray(list_imgs, dtype=theano.config.floatX),  # @UndefinedVariable
#                              borrow=borrow)
# 
#     shared_y = theano.shared(np.asarray([1, 0], dtype=theano.config.floatX),  # @UndefinedVariable
#                              borrow=borrow)
# 
#     return shared_x, T.cast(shared_y, 'int32')

def test_validation(twelve_net_state, batch_size=50):

    train_set_x, train_set_y = get_train_data()
    valid_set_x, valid_set_y = get_valid_data()
    test_set_x, test_set_y = get_train_data()


    # compute number of minibatches for training, validation and testing
    num_train_samples = train_set_x.get_value(borrow=True).shape[0]
    num_valid_samples = valid_set_x.get_value(borrow=True).shape[0]
    num_test_samples = test_set_x.get_value(borrow=True).shape[0]

    n_train_batches = num_train_samples/batch_size
    n_valid_batches = num_valid_samples/batch_size
    n_test_batches = num_test_samples/batch_size


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 3, 13, 13))
    #layer0_input = x

    net = twelve_net(layer0_input,batch_size,twelve_net_state)


    errors = net.log_regression_layer.errors( y)
    validate_model = theano.function(
        [index],
        errors,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validation_losses = [validate_model(i) for i
                                in xrange(n_valid_batches)]
    this_validation_loss = np.mean(validation_losses)
    print('validation error %f %%' %
            (this_validation_loss * 100.))


def check_for_image():

    import skimage
    import skimage.data
    import skimage.util


    img = skimage.data.lena()
    #img = io.imread("data/originalPics/2002/07/19/big/img_130.jpg")
    
    #img = io.imread("data/train_set/dataset/13/train/faces/00027998.png")
    
    img = rgb2gray(img)
    
    img = img_as_ubyte(img)  
    img=img[:,:,np.newaxis]
    

    #img = imread("data/processed_images/13_train_set_aflw/train/faces/001111.jpg")

    for mul in xrange(3, 20):

        im = resize(img, (mul*10, mul*10))
        im = img_as_ubyte(im) 
        arr = skimage.util.view_as_windows(im, (13, 13, NUM_CHANNELS), step=1)
        f = file(LOAD_STATE_FROM_FILE, 'rb')
        obj = pickle.load(f)
        f.close()

        arr = np.rollaxis(arr, 5, 3)

        borrow = True
        shared_x = theano.shared(np.asarray(arr, dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)

        iT = T.lscalar()
        jT = T.lscalar()

        x = T.tensor3("x")
        layer0_input = x.reshape((1, NUM_CHANNELS, 13, 13))

        net = twelve_net(layer0_input, None, relu, obj)
        prediction = net.log_regression_layer.y_pred
        py_x = net.log_regression_layer.p_y_given_x

        test_model = theano.function(
            [iT, jT],
            [prediction, py_x, layer0_input],
            givens={
                x: shared_x[iT, jT, 0, :, :, :]
            }
        )

        rows = arr.shape[0]
        cols = arr.shape[1]

        count = 0

        faces = []
        
        fig,axarr = plt.subplots(1,2)
        confidence_map = np.zeros((rows,cols))
        
        for i in xrange(rows):
            for j in xrange(cols):
                [y,p_y_given_x,f] = test_model(i,j)
                f=f.reshape(NUM_CHANNELS,13,13)
                f = np.rollaxis(f,0,3)
                f = f[:,:,0]
                confidence_map[i,j] = p_y_given_x[0,1]
                
#                 plt.imshow(f,cmap = "Greys_r")
#                 plt.show()



                if y == 1:
                    count += 1
                    faces.append([i,j])
                    print i,j

        print ("Check")
        # hack for gray
        im = im[:,:,0]
        
        confidence_map=np.pad(confidence_map, 6,'constant')
        axarr[1].imshow(confidence_map,cmap = "Greys_r")
        axarr[0].imshow(im,cmap = "Greys_r")
        img_desc = plt.gca()
        
        
        #plt.imshow(im,cmap = "Greys_r")
        #img_desc = plt.gca()

        for point in faces:
            topleftx = point[1] 
            toplefty = point[0] 


            rect = patches.Rectangle(
                (topleftx, toplefty),
                13,
                13,
                fill=False,
                color='c'
            )

            axarr[0].add_patch(rect)

        print count
        #fig_name = "lena_" + str(mul)+".png"
        #plt.savefig(fig_name, bbox_inches='tight')
        plt.show()

def evaluate_trained_net_on_patches(state,folder="data/train_set/dataset/13/train/nonfaces/*.jpg"):
    img_collection = imread_collection(folder)
    
    arr_imgs = concatenate_images(img_collection)
    arr_imgs=arr_imgs[:,:,:,np.newaxis]
    arr_imgs = np.rollaxis(arr_imgs, 3, 1)
    
    
    #arr_imgs = arr_imgs[0:10,:,:,:]
    borrow = True
    shared_x = theano.shared(np.asarray(arr_imgs, dtype=theano.config.floatX),  # @UndefinedVariable
                             borrow=borrow)
    
   
    num_arr_imgs= arr_imgs.shape[0]
    iT = T.lscalar()
    x = T.tensor3("x")
    layer0_input = x.reshape((1, NUM_CHANNELS, 13, 13))

    net = twelve_net(layer0_input, None, relu, state)
    prediction = net.log_regression_layer.y_pred
    py_x = net.log_regression_layer.p_y_given_x

    test_model = theano.function(
            [iT],
            [prediction, py_x, layer0_input],
            givens={
                x: shared_x[iT, :, :, :]
            }
    )
    
    min_py_x = 99999;
    max_py_x = 0
    num_faces = 0
    for i in xrange(num_arr_imgs):
        [out_predict,out_py_x,out_face]=test_model(i)
        
        if out_py_x[0,1] < min_py_x :
            min_py_x = out_py_x[0,1]
        
        if out_py_x[0,1] > max_py_x :
            max_py_x = out_py_x[0,1]
            
        if out_predict ==1:
            num_faces =num_faces +1
    
    
    print "Num faces: %d" %(num_faces) 
    print "min_py_x: %f" %(min_py_x) 
    print "max_py_x: %f" %(max_py_x) 
    print "Num images evaluated %d" %(num_arr_imgs)
    
            
         
    
    
    
    
    
    
# def check_image_from_training():
# 
#     import skimage
#     import skimage.data
#     import skimage.util
# 
# 
#     #img = imread("data/originalPics/2002/07/19/big/img_372.jpg")
#     #img = skimage.data.lena()
# 
#     #im = imread("data/processed_images/13_train_set_aflw/train/faces/001111.jpg")
#     im = imread("data/processed_images/13_train_set_aflw/train/faces/001111.jpg")
# 
#     arr = skimage.util.view_as_windows(im, (13, 13, 3), step=1)
#     f = file(LOAD_STATE_FROM_FILE, 'rb')
#     obj = pickle.load(f)
#     f.close()
# 
#     arr = np.rollaxis(arr, 5, 3)
# 
#     borrow = True
#     shared_x = theano.shared(np.asarray(arr, dtype=theano.config.floatX),  # @UndefinedVariable
#                              borrow=borrow)
# 
#     iT = T.lscalar()
#     jT = T.lscalar()
# 
#     x = T.tensor3("x")
#     layer0_input = x.reshape((1, 3, 13, 13))
# 
#     net = twelve_net(layer0_input, None, relu, obj)
#     prediction = net.log_regression_layer.y_pred
#     py_x = net.log_regression_layer.p_y_given_x
# 
#     test_model = theano.function(
#         [iT, jT],
#         [prediction, py_x, layer0_input],
#         givens={
#             x: shared_x[iT, jT, 0, :, :, :]
#         }
#     )
# 
#     rows = arr.shape[0]
#     cols = arr.shape[1]
# 
#     count = 0
# 
#     faces = []
# 
#     for i in xrange(rows):
#         for j in xrange(cols):
#             [y,p_y_given_x,f] = test_model(i,j)
# 
#             if y == 1:
#                 count += 1
#                 faces.append([i,j])
#                 print i,j
# 
#     print ("Check")
#     plt.imshow(im)
#     img_desc = plt.gca()
# 
#     for point in faces:
#         topleftx = point[1]
#         toplefty = point[0]
# 
#         rect = patches.Rectangle(
#             (topleftx, toplefty),
#             13,
#             13,
#             fill=False,
#             color='c'
#         )
# 
#         img_desc.add_patch(rect)
# 
#     print count
# 
#     plt.show()

    
def experiment():
    #collection = ImageCollection('data/test/*.jpg')

    #evaluate_12net()
    f = file(LOAD_STATE_FROM_FILE, 'rb')
    obj = pickle.load(f)
    f.close()
    
    test_validation(obj)
    
def experiment_train():
    #collection = ImageCollection('data/test/*.jpg')

    evaluate_12net()

# def check2():
#
#     import skimage
#     import skimage.data
#     import skimage.util
#
#     img = imread("data/originalPics/2002/07/19/big/img_372.jpg")
#
#     f = file(LOAD_STATE_FROM_FILE, 'rb')
#     obj = pickle.load(f)
#     f.close()
#
#     arr = np.rollaxis(arr, 5, 3)
#
#     borrow = True
#     shared_x = theano.shared(np.asarray(arr, dtype=theano.config.floatX),
#                              borrow=borrow)

def visualize(fig,axarr,W):
    W_rolled = np.rollaxis(W, 1, 4)
    num_filters = W_rolled.shape[0]
    #fig = plt.figure()

    #fig,axarr = plt.subplots(4,4)
    Wmin = W_rolled.min();
    Wmax = W_rolled.max();
    
    
    
    for i in xrange(num_filters):
        row_no = i / 4
        col_no = i % 4
        #fig.add_subplot(4,4,i+1)
        conv_filt = W_rolled[i,:,:,:]
        #data_row = conv_filt.flatten()
#       data_row = data_row - data_row.min()
#       data_row = data_row/data_row.max()
        #data_row = data_row * 255
        conv_filt = conv_filt - Wmin;
        #Wmax = W_rolled.max();
        conv_filt = conv_filt / conv_filt.max()
        img = conv_filt *255
        
        #img = data_row.reshape(3,3,1)#.astype('uint8')
        img = img[:,:,0]
        #fig.figimage(img,(i+1)*3,0)
        #pil_img = Image.fromarray(img)
        #s = "filte" + str(i) + ".jpg"
        #pil_img.save(s)
        #plt.imshow(pil_img)
        #pil_img.show()
        axarr[row_no,col_no].imshow(img,interpolation = 'none',cmap = "Greys_r")
        #plt.show()
    
    #plt.show()
    fig.canvas.draw()
    
#     a = np.zeros((3,3,3))
#     a[0,0,0] = 255
#     a[0,0,1] = 255
#     axarr[0,0].imshow(a)
#     fig.canvas.draw()
    
    
    
    print "Visualizing"



def visualize_filter():
    f = file(LOAD_STATE_FROM_FILE, 'rb')
    obj = pickle.load(f)
    f.close()
    W = obj[0];
    W_rolled = np.rollaxis(W, 1, 4)
    num_filters = W_rolled.shape[0]
    #fig = plt.figure()
    plt.ion()
    fig,axarr = plt.subplots(4,4)
    for i in xrange(num_filters):
        row_no = i / 4
        col_no = i % 4
        #fig.add_subplot(4,4,i+1)
        conv_filt = W_rolled[i,:,:,:]
        data_row = conv_filt.flatten()
        data_row = data_row - data_row.min()
        data_row = data_row/data_row.max()
        data_row = data_row * 255
        img = data_row.reshape(3,3,3)#.astype('uint8')
        #fig.figimage(img,(i+1)*3,0)
        #pil_img = Image.fromarray(img)
        #s = "filte" + str(i) + ".jpg"
        #pil_img.save(s)
        #plt.imshow(pil_img)
        #pil_img.show()
        axarr[row_no,col_no].imshow(img,interpolation = 'none')
        #plt.show()
    
    #plt.show()
    fig.canvas.draw()
    
    a = np.zeros((3,3,3))
    a[0,0,0] = 255
    a[0,0,1] = 255
    axarr[0,0].imshow(a)
    fig.canvas.draw()
    
    
    
    print "Visualizing"

def call_evalate_trained_net():
    f = file(LOAD_STATE_FROM_FILE, 'rb')
    obj = pickle.load(f)
    f.close()    
        
    evaluate_trained_net_on_patches(state = obj)
        
if __name__ == '__main__':
    check_for_image()
    #experiment_train()
    
    #call_evalate_trained_net()
    


