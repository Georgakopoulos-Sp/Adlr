
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/home/legolas/CNN_libs/caffe_mike/caffe/python/caffe'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

plt.rcParams['image.cmap'] = 'gray'

import caffe

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data,name='res/aaa.png', padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
#    plt.draw()
    plt.savefig(name, dpi=1000)


#net = caffe.Net( net_def_proto , weights )
#net = caffe.Net( '/home/legolas/CNN_libs/mycaffe_test/examples/cifar10/cifar10_quick_train_test.prototxt' , '/home/legolas/CNN_libs/mycaffe_test/examples/cifar10/cifar10_quick_iter_4000.caffemodel', caffe.TEST )
net = caffe.Net( '/home/legolas/Desktop/project/challeng_iakovidis/journals/neurocomputing/code/try3/barretts_quick_train_test.prototxt' , '/home/legolas/Desktop/project/challeng_iakovidis/journals/neurocomputing/code/try3/snapshots/snapshot_Msgd_1_iter_4000.caffemodel', caffe.TEST )
#net = caffe.Net( '/home/legolas/CNN_libs/mycaffe_test/examples/cifar10/cifar10_quick_train_test.prototxt' , caffe.TEST )
caffe.set_device(0)
caffe.set_mode_gpu()

label = net.blobs['accuracy'].data
print label
res = net.forward()
print res

label = net.blobs['label'].data
print label
print [(k, v.data.shape) for k, v in net.blobs.items()]
print [(k, v[0].data.shape) for k, v in net.params.items()]

for i in range(0, 49):
	data_name = 'data'
	data_name +=str(i)
	data_name = data_name +'_label_'+str(label[i])
	data_name +='.png'
	data_var = net.blobs['data'].data[i]
	plt.imshow(data_var[1,:320, :320])
	plt.savefig('res/'+data_name, dpi=100)

	name_conv = 'conv1_'
	name_conv += str(i)
	name_conv = 'res/'+name_conv
	name_conv = name_conv + '_label_'+str(label[i])
	name_conv +='.png'
	conv_data = net.blobs['conv1'].data[i]
	vis_square(conv_data, name=name_conv, padval=0.5)

	name_conv = 'conv2_'
	name_conv += str(i)
	name_conv = 'res/'+name_conv
	name_conv = name_conv + '_label_'+str(label[i])
	name_conv +='.png'
	conv_data = net.blobs['conv2'].data[i]
	vis_square(conv_data, name=name_conv, padval=0.5)
	
	name_conv = 'conv3_'
	name_conv += str(i)
	name_conv = 'res/'+name_conv
	name_conv = name_conv + '_label_'+str(label[i])
	name_conv +='.png'
	conv_data = net.blobs['conv3'].data[i]
	vis_square(conv_data, name=name_conv, padval=0.5)
	
	


#filters = net.params['conv1'][0].data
#print filters
#test = filters.transpose(0,2,3,1)
#test = filters[1,1, :5, :5]
#print filters.shape
#print test.shape
#vis_square(filters.transpose(0, 2, 3, 1))
#plt.imshow(test)
#plt.savefig('tessstttyyy.png', dpi=1000)
#feat = net.blobs['conv2'].data[2]
#print feat.shape
#thefirst = net.blobs['data'].data[2]
#print thefirst.shape
#label = net.blobs['label'].data
#print label
#print net.blobs['data'].data.shape
#plt.imshow(thefirst[1,:320, :320])
#plt.savefig('data_2.png', dpi=100)
#plt.imshow(feat[1,1,:32,:32])
#plt.savefig('tessstttyyy.png', dpi=1000)
#vis_square(feat, padval=0.5)
