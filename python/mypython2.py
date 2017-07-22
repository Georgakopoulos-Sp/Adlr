import numpy as np
#import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/home/legolas/CNN_libs/caffe_mike/caffe/python/caffe'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

#net = caffe.Net( net_def_proto , weights )
net = caffe.Net( '/home/legolas/CNN_libs/mycaffe_test/examples/cifar10/cifar10_quick_train_test.prototxt' , '/home/legolas/CNN_libs/mycaffe_test/examples/cifar10/cifar10_quick_iter_4000.caffemodel', caffe.TEST )
#net = caffe.Net( '/home/legolas/CNN_libs/mycaffe_test/examples/cifar10/cifar10_quick_train_test.prototxt' , caffe.TEST )
caffe.set_device(0)
caffe.set_mode_gpu()
#net.set_phase_test()
net.blobs['data'].data[...]=net.blobs['data'].data[1,:,:,:]
out = net.forward()
print net.blobs['data'].data.shape
print net.blobs['data'].num
print out
print out['loss']
print out['accuracy']
print net.params
