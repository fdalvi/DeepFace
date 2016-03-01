import numpy as np
import caffe
import os

import util

DATA_PATH = '../data/dev_set/images_cropped/'
WEIGHTS_PATH = 'vgg_face_caffe/VGG_FACE.caffemodel'
ARCH_PATH = 'DeepFaceNet.prototxt'
SOLVER_PATH = 'solver.prototxt'

def evaluate():
	caffe.set_mode_gpu()
	assert os.path.exists(WEIGHTS_PATH)
	assert os.path.exists(ARCH_PATH)

	net = caffe.Net(ARCH_PATH,
	                WEIGHTS_PATH,
	                caffe.TEST)

	out = net.forward()
	# Old head
	# print net.blobs['prob'].data
	# print out['argmax']
	# 
	# Old head without ARGMAX
	# for i in xrange(len(out['prob'])):
		# print("Predicted class is #{}.".format(out['prob'][i].argmax()))
	
	print out['loss-1']
	print out['loss-2']
	# print out['argmax-1']
	# print out['argmax-2']
	# print out['label-1']
	# print out['label-2']
	print "done."

def train():
	if not os.path.exists('snapshots'):
		os.makedirs('snapshots')
	caffe.set_mode_gpu()
	solver = caffe.get_solver(SOLVER_PATH)
	print "Loading old weights..."
	solver.net.copy_from(WEIGHTS_PATH)
	print "Stepping..."
	solver.solve()
	print "done."

def main():
	train()

if __name__ == '__main__':
	main()