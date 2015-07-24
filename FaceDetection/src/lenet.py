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
from matplotlib.image import imread
from skimage.transform._warps import resize
from skimage.io._io import imread_collection

from matplotlib import pyplot as plt
from matplotlib import patches
from relu import relu

from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray


SAVE_STATE_TO_FILE = "test_tanh.save"
LOAD_STATE_FROM_FILE = "src/weights/lenet_original_weights.save"

SAVE_MEAN_TO_FILE = "test_tanh_mean.save"
LOAD_MEAN_TO_FILE = "src/weights/lenet_original_mean.save"

# settings for LBP
lbp_radius = 3
lbp_n_points = 8 * lbp_radius
lbp_method = 'uniform'

def save_list_of_weights(filename, weights_list):
    with open(filename, 'wb') as output:
        pickle.dump(weights_list, output, pickle.HIGHEST_PROTOCOL)

def save_mean_val(filename, mean_val):
    with open(filename, 'wb') as output:
        pickle.dump(mean_val, output, pickle.HIGHEST_PROTOCOL)

class twelve_net():

    def __init__(self, input, batch_size, activation, state=None):

        rng = np.random.RandomState(23455)

        img_size = 28
        img_channels = 1

        conv_filter_size_1 = 5
        conv_filter_size_2 = 5
        conv_filter_stride = 1 #hard coded
        conv_filter_depth_1 = 20
        conv_filter_depth_2 = 50

        ## Not used becusee it is hardcoded  inside le-net
        pool_filter_size = 3
        pool_filter_stride = 2

        conv_pool_output_size = 5 ## 10

        fullyconnected_output_size = 500

        self.input = input

        if state is None:
            conv_pool_layer_state_1 = None
            conv_pool_layer_state_2 = None
            fully_connected_layer_state = None
            log_regression_layer_state = None
        else:
            conv_pool_layer_state_1 = state[0:2]
            conv_pool_layer_state_2 = state[2:4]
            fully_connected_layer_state = state[4:6]
            log_regression_layer_state = state[6:8]


        self.conv_pool_layer_1 = LeNetConvPoolLayer(
            rng,
            input=input,
            image_shape=(batch_size, img_channels, img_size, img_size),
            filter_shape=(conv_filter_depth_1, img_channels, conv_filter_size_1, conv_filter_size_1),
            poolsize=(2, 2),
            activation=activation,
            state = conv_pool_layer_state_1
        )

        self.conv_pool_layer_2 = LeNetConvPoolLayer(
            rng,
            input=self.conv_pool_layer_1.output,
            image_shape=(batch_size, conv_filter_depth_1, 12, 12),
            filter_shape=(conv_filter_depth_2, conv_filter_depth_1, conv_filter_size_2, conv_filter_size_2),
            poolsize=(2, 2),
            activation=activation,
            state = conv_pool_layer_state_2
        )

        self.fullyconnected_layer = HiddenLayer(
            rng,
            input=self.conv_pool_layer_2.output.flatten(2),
            n_in= 4 * 4 * conv_filter_depth_2,
            n_out=fullyconnected_output_size,
            activation=activation,
            state = fully_connected_layer_state
        )

        self.log_regression_layer = LogisticRegression(input=self.fullyconnected_layer.output,
                                            n_in=fullyconnected_output_size,
                                            n_out=2, state=log_regression_layer_state)

        self.params = (self.conv_pool_layer_1.params +
                       self.conv_pool_layer_2.params +
                       self.fullyconnected_layer.params +
                       self.log_regression_layer.params)

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
        variance = np.sqrt(2.0/fan_in)

        print variance

        if state is None:
        # initialize weights with random weights

            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W = theano.shared(
                np.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    #rng.normal(loc=0, scale=variance, size=filter_shape),
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

        conv_layer_stride = 1
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            subsample=(conv_layer_stride, conv_layer_stride)
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True#,
            #st = (1, 1)
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

def evaluate_12net(learning_rate=0.01, n_epochs=300,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=250):
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

    image_size = 28
    image_depth = 1

    train_set_x, train_set_y = get_train_data()
    valid_set_x, valid_set_y = get_valid_data()
    test_set_x, test_set_y = get_train_data()

    # Compute and save mean value
    train_set_x_val = train_set_x.get_value(borrow=True)
    valid_set_x_val = valid_set_x.get_value(borrow=True)
    test_set_x_val = test_set_x.get_value(borrow=True)

    mean_train_set_val = compute_mean_from_dataset(train_set_x_val)

    train_set_x_val_centered = train_set_x_val - mean_train_set_val
    valid_set_x_val_centered = valid_set_x_val - mean_train_set_val
    test_set_x_val_centered = test_set_x_val - mean_train_set_val

    train_set_x.set_value(train_set_x_val_centered, borrow=True)
    valid_set_x.set_value(valid_set_x_val_centered, borrow=True)
    test_set_x.set_value(test_set_x_val_centered, borrow=True)

    mean_val_to_save = mean_train_set_val.reshape((image_depth, image_size, image_size))
    mean_val_to_save = np.rollaxis(mean_val_to_save, 0, 3)

    save_mean_val(SAVE_MEAN_TO_FILE, mean_val_to_save)

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
    layer0_input = x.reshape((batch_size, image_depth, image_size, image_size))
    #layer0_input = x

    net = twelve_net(layer0_input, batch_size, T.tanh)

    cost = net.log_regression_layer.negative_log_likelihood(y)
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

    # Get the actual weights values from symbolic var.
    # It's done in order to save them later.
    best_params = map(lambda x: x.eval(), net.params)

    # compute zero-one loss on validation set
    validation_losses = [validate_model(i) for i
                                in xrange(n_valid_batches)]

    this_validation_loss = np.mean(validation_losses)
    print('Pre training  validation error %f %%' %
            (this_validation_loss * 100.))

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
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

def convert_image_stack_to_lbp(img_arr):

    lbp_stack = np.zeros((img_arr.shape[0], img_arr.shape[1], img_arr.shape[2], 1))
    img_amount = img_arr.shape[0]

    for img_count in xrange(img_amount):

        lbp_stack[img_count, :, :, 0] = local_binary_pattern(rgb2gray(img_arr[img_count, :, :, :]),
                                                             lbp_n_points, lbp_radius, lbp_method)

    return lbp_stack

def convert_image_stack_to_gray(img_arr):

    gray_stack = np.zeros((img_arr.shape[0], img_arr.shape[1], img_arr.shape[2], 1))
    img_amount = img_arr.shape[0]

    for img_count in xrange(img_amount):

        if len(img_arr.shape) > 3:
            gray_img = rgb2gray(img_arr[img_count, :, :, :])
        else:
            gray_img = img_arr[img_count, :, :]

        gray_stack[img_count, :, :, 0] = gray_img

    return gray_stack

def prepare_data(faces_collection, bkgs_collection):

    # To get a subset:
    # arr_faces = concatenate_images(faces_collection)[:5000, ...]
    arr_faces = concatenate_images(faces_collection)
    arr_faces = convert_image_stack_to_gray(arr_faces)

    arr_faces = np.rollaxis(arr_faces, 3, 1)
    arr_faces = arr_faces
    num_face_imgs = arr_faces.shape[0]
    arr_faces = arr_faces.reshape((arr_faces.shape[0], -1))
    out_faces = np.ones(arr_faces.shape[0])

    arr_bkgs = concatenate_images(bkgs_collection)
    arr_bkgs = convert_image_stack_to_gray(arr_bkgs)
    arr_bkgs = np.rollaxis(arr_bkgs, 3, 1)
    arr_bkgs = arr_bkgs.reshape((arr_bkgs.shape[0], -1))
    arr_bkgs = arr_bkgs
    out_bkgs = np.zeros(arr_bkgs.shape[0])

    test_set = np.concatenate((arr_faces, arr_bkgs))
    labels = np.concatenate((out_faces, out_bkgs))

    arr_indexes = np.random.permutation(test_set.shape[0])

    shuffled_test_set = test_set[arr_indexes]
    shuffled_labels = labels[arr_indexes].flatten()

    borrow = True
    shared_x = theano.shared(np.asarray(shuffled_test_set, dtype=theano.config.floatX),
                             borrow=borrow)

    shared_y = theano.shared(np.asarray(shuffled_labels, dtype=theano.config.floatX),
                             borrow=borrow)

    return shared_x, T.cast(shared_y, 'int32')

def get_valid_data():

    img_faces = imread_collection("data/processed_images/28_28_big_new/validation/faces/*.png")

    img_bkgs = imread_collection("data/processed_images/28_28_big_new/validation/nonfaces/*.jpg")

    x, y = prepare_data(img_faces, img_bkgs)

    return x, y

def get_train_data():

    img_faces = imread_collection("data/processed_images/28_28_big_new/train/faces/*.png")

    img_bkgs = imread_collection("data/processed_images/28_28_big_new/train/nonfaces/*.jpg")

    x, y = prepare_data(img_faces, img_bkgs)

    return x, y

def compute_mean_from_dataset(data_matrix):

    return np.mean(data_matrix, axis=0)

def check_for_image():

    import skimage
    import skimage.data
    import skimage.util
    from skimage.color import rgb2gray

    img = rgb2gray(imread("data/originalPics/2002/07/19/big/img_372.jpg"))
    #img = rgb2gray(skimage.data.lena())
    #img = imread("data/processed_images/13_train_set_aflw/train/faces/001111.jpg")

    # Load mean value from file.
    f = file(LOAD_MEAN_TO_FILE, 'rb')
    mean_val = pickle.load(f)
    f.close()

    # Scale search
    for mul in xrange(3, 20):

        img_orig = resize(img, (mul*10, mul*10))
        im = img_orig[..., np.newaxis]

        arr = skimage.util.view_as_windows(im, (28, 28, 1), step=1)

        arr = arr - mean_val

        f = file(LOAD_STATE_FROM_FILE, 'rb')
        obj = pickle.load(f)
        f.close()

        arr = np.rollaxis(arr, 5, 3)

        borrow = True
        shared_x = theano.shared(np.asarray(arr, dtype=theano.config.floatX),
                                 borrow=borrow)

        iT = T.lscalar()
        jT = T.lscalar()

        x = T.tensor3("x")
        layer0_input = x.reshape((1, 1, 28, 28))

        net = twelve_net(layer0_input, None, T.tanh, obj)
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

        for i in xrange(rows):
            for j in xrange(cols):
                [y,p_y_given_x,f] = test_model(i, j)
                f=f.reshape(1, 28, 28)
                f = np.rollaxis(f, 0, 3)
                #plt.imshow(f)
                #plt.show()

                if y == 1:
                    count += 1
                    faces.append([i, j])
                    print i, j

        print ("Check")
        plt.imshow(img_orig)
        plt.set_cmap('gray')
        img_desc = plt.gca()

        for point in faces:
            topleftx = point[1]
            toplefty = point[0]

            rect = patches.Rectangle(
                (topleftx, toplefty),
                28,
                28,
                fill=False,
                color='c'
            )

            img_desc.add_patch(rect)

        print count

        plt.show()

def check_image_from_training():

    import skimage
    import skimage.data
    import skimage.util

    #img = imread("data/originalPics/2002/07/19/big/img_372.jpg")
    #img = skimage.data.lena()

    f = file(LOAD_MEAN_TO_FILE, 'rb')
    mean_val = pickle.load(f)
    f.close()

    img_orig = rgb2gray(imread("data/processed_images/28_train_set_usual/train/nonfaces/000001.jpg"))

    im = img_orig[..., np.newaxis]

    im = im - mean_val

    # for mul in xrange(3, 20):
    #
    #     im = resize(img, (mul*10, mul*10))

    arr = skimage.util.view_as_windows(im, (28, 28, 1), step=1)

    f = file(LOAD_STATE_FROM_FILE, 'rb')
    obj = pickle.load(f)
    f.close()

    arr = np.rollaxis(arr, 5, 3)

    borrow = True
    shared_x = theano.shared(np.asarray(arr, dtype=theano.config.floatX),
                             borrow=borrow)

    iT = T.lscalar()
    jT = T.lscalar()

    x = T.tensor3("x")
    layer0_input = x.reshape((1, 1, 28, 28))

    net = twelve_net(layer0_input, None, T.tanh, obj)
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

    for i in xrange(rows):
        for j in xrange(cols):
            [y,p_y_given_x,f] = test_model(i,j)

            if y == 1:
                count += 1
                faces.append([i,j])
                print i,j

    print ("Check")
    plt.imshow(img_orig)
    plt.set_cmap('gray')
    img_desc = plt.gca()

    for point in faces:
        topleftx = point[1]
        toplefty = point[0]

        rect = patches.Rectangle(
            (topleftx, toplefty),
            28,
            28,
            fill=False,
            color='c'
        )

        img_desc.add_patch(rect)

    print count

    plt.show()


if __name__ == '__main__':
    #check_image_from_training()
    check_for_image()
    #evaluate_12net()



