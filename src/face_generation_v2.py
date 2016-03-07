import numpy as np
import os
import random
import caffe
import matplotlib.pyplot as plt
import util
import sklearn
import sys
import cPickle as cp

DATA_PATH = '../data/eval_set/images_cropped/'
WEIGHTS_PATH = './snapshots/_iter_42000.caffemodel'
SOLVER_PATH = './DeepFaceNetDeploy.prototxt'
LAYER = 'fc6'
OUTPUT_PATH = './'
# NUM_ATTRIBUTES = 73

def main(): 
	# def extract_activations(layer, data_path, weights_path, solver_path, output_path, batch_size=25):
	if len(sys.argv) == 1: 
		mode = 'compute'
	else: 
		mode = sys.argv[1]

	if mode == 'extract':
		util.extract_activations(LAYER, DATA_PATH, WEIGHTS_PATH, SOLVER_PATH, OUTPUT_PATH)

	with open(OUTPUT_PATH + 'outs-%s/image_list.dat'%(LAYER), 'rb') as fp:
		images = cp.load(fp)
	labels = util.get_attributes('../data/pubfig_attributes.txt', images)
	# print 'labels shape', labels.shape
	labels = labels.as_matrix()
	# print labels[0,:]
	activations = np.load(OUTPUT_PATH + 'outs-%s/blob.dat.npy'%(LAYER))
	

	num_attributes = labels.shape[1]
	num_features = np.prod(activations.shape)/activations.shape[0]
	means = np.zeros((num_attributes, num_features))
	# covs = np.zeros((num_attributes, num_features, num_features))
	vars_ = np.zeros((num_attributes, num_features))
	for i in xrange(labels.shape[1]):
		print 'Iteration %d...'%(i+1)
		selected_examples = activations[(labels[:,i] > 0.5),:]
		selected_examples = selected_examples.reshape((selected_examples.shape[0], -1))
		means[i, :] = np.mean(selected_examples, axis=0)
		# covs[i, :, :] = np.cov(selected_examples, rowvar=0)
		vars_[i,:] = np.var(selected_examples, axis=0)

	np.save('means-%s'%(LAYER), means)
	np.save('vars-%s'%(LAYER), vars_)

		# print selected_examples.shape

	# mixture = sklearn.mixture.GMM(n_components=73, n_iter=5, n_init=5, params='w', init_params='w')

def extract():
	return util.extract_activations(LAYER, DATA_PATH, WEIGHTS_PATH, SOLVER_PATH, OUTPUT_PATH)

if __name__ == '__main__':
	main()
