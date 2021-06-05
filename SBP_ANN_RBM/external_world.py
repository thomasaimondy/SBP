import pickle as cpickle 
import gzip
import numpy as np
import os
import theano
import theano.tensor as T
import theano.tensor.extra_ops
import scipy.io as sio
import pdb
class External_World(object):

    def __init__(self,tasktype):
        
        if tasktype is 'mnist':
            # pdb.set_trace()
            dir_path = os.path.dirname(os.path.abspath(__file__))
            path = dir_path+os.sep+"mnist.pkl.gz"

            # DOWNLOAD MNIST DATASET
            if not os.path.isfile(path):
                import urllib.request
                origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
                print('Downloading data from %s' % origin)
                urllib.request.urlretrieve(origin, path)

            # LOAD MNIST DATASET
            f = gzip.open(path, 'rb')
            # (train_x_values, train_y_values), (valid_x_values, valid_y_values), (test_x_values, test_y_values) = cpickle.load(f)
            (train_x_values, train_y_values), (valid_x_values, valid_y_values), (test_x_values, test_y_values) = cpickle.load(f,encoding='bytes')
            f.close()

            # pdb.set_trace()
            # CONCATENATE TRAINING, VALIDATION AND TEST SETS
            x_values = list(train_x_values) + list(valid_x_values) + list(test_x_values)
            y_values = list(train_y_values) + list(valid_y_values) + list(test_y_values)
            self.x = theano.shared(np.asarray(x_values, dtype=theano.config.floatX), borrow=True)
            self.y = T.cast(theano.shared(np.asarray(y_values, dtype=theano.config.floatX), borrow=True), 'int32')
            self.y_onehot = T.extra_ops.to_one_hot(self.y, 10)
            self.size_dataset = len(x_values)
            # pdb.set_trace()

        if tasktype is 'nettalk':
            mat_fname = 'nettalk_small.mat'
            mat_contents = sio.loadmat(mat_fname)
            train_x_values = mat_contents['train_x']
            train_y_values = mat_contents['train_y']
            valid_x_values = mat_contents['test_x']
            valid_y_values = mat_contents['test_y']
            test_x_values = mat_contents['test_x']
            test_y_values = mat_contents['test_y']
            x_values = list(train_x_values) + list(valid_x_values) + list(test_x_values)
            y_values = list(train_y_values) + list(valid_y_values) + list(test_y_values)
            self.x =        theano.shared(np.asarray(x_values, dtype=theano.config.floatX), borrow=True)
            self.y = T.cast(theano.shared(np.asarray(y_values, dtype=theano.config.floatX), borrow=True), 'int32')
            self.y_onehot = self.y
            self.size_dataset = len(x_values)
        

        if tasktype is 'gesture':
            mat_fname = 'DVS_gesture_100.mat'
            mat_contents = sio.loadmat(mat_fname)
            train_x_values = mat_contents['train_x_100'].reshape(1176,102400)
            train_y_values = mat_contents['train_y']
            valid_x_values = mat_contents['test_x_100'].reshape(288,102400)
            valid_y_values = mat_contents['test_y']
            test_x_values = mat_contents['test_x_100'].reshape(288,102400)
            test_y_values = mat_contents['test_y']
            # pdb.set_trace()
            # pdb.set_trace()
            x_values = list(train_x_values) + list(valid_x_values) + list(test_x_values)
            y_values = list(train_y_values) + list(valid_y_values) + list(test_y_values)
            self.x =        theano.shared(np.asarray(x_values, dtype=theano.config.floatX), borrow=True)
            self.y = T.cast(theano.shared(np.asarray(y_values, dtype=theano.config.floatX), borrow=True), 'int32')
            # pdb.set_trace()
            self.y_onehot = self.y
            self.size_dataset = len(x_values)
            
