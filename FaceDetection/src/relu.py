'''
Created on 11-Jul-2015

@author: amilgeorge
'''

import theano.tensor as T
#######################
#Rectified Linear Unit#
#######################
def relu(x):
    return T.switch(x<0, 0, x)