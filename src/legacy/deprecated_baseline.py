import numpy as np
import caffe
import os

import util

DATA_PATH = '../data/dev_set/images_cropped/'

def main():
	caffe.set_mode_gpu()

	images = ['Barack_Obama_441', 'Ali_Landry_78', 'Ben_Stiller_69']
	images_with_path = [DATA_PATH + i + '.jpg' for i in images]

	weights_path = 'vgg_face_caffe/VGG_FACE.caffemodel'
	assert os.path.exists(weights_path)

	solver_path = 'vgg_face_caffe/VGG_FACE_deploy.prototxt'
	assert os.path.exists(solver_path)

	# caffe.set_mode_cpu()
	net = caffe.Net(solver_path,
	                weights_path,
	                caffe.TEST)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))

	# Reshape to match batch size
	data_blob_shape = net.blobs['data'].data.shape
	data_blob_shape = list(data_blob_shape)
	net.blobs['data'].reshape(len(images), data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

	net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images_with_path)

	out = net.forward()
	print out
	print("Predicted class is #{}.".format(out['prob'][0].argmax()))
	print("Predicted class is #{}.".format(out['prob'][1].argmax()))
	print("Predicted class is #{}.".format(out['prob'][2].argmax()))


if __name__ == '__main__':
	main()