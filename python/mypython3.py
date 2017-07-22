
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
accuracy=0
myrange = 100
for i in range(myrange):
	out = net.forward()
	accuracy +=out['accuracy']
#	print net.blobs['accuracy'].data
	print out['accuracy']
print accuracy
