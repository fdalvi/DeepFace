import numpy as np
import os
import random
import caffe
import matplotlib.pyplot as plt
import util
import sklearn
import sys
import cPickle as cp

from gmm import GMM

DATA_PATH = '../data/eval_set/images_cropped/'
WEIGHTS_PATH = './snapshots/_iter_42000.caffemodel'
SOLVER_PATH = './DeepFaceNetDeploy.prototxt'
LAYER = 'conv3_1'
# NUM_ATTRIBUTES = 73
NUM_SAMPLES = 100
OUTPUT_PATH = './trial_%s_%d/'%(LAYER, NUM_SAMPLES)

LAYER_SIZES = {
		  "conv1_1": (224, 224, 64), 
		  "conv1_2": (224, 224, 64), 
		  "conv2_1": (112, 112, 128), 
		  "conv2_2": (112, 112, 128), 
		  "conv3_1": (56, 56, 256), 
		  "conv3_2": (56, 56, 256), 
		  "conv3_3": (56, 56, 256), 
		  "conv4_1": (28, 28, 512), 
		  "conv4_2": (28, 28, 512), 
		  "conv4_3": (28, 28, 512), 
		  "conv5_1": (14, 14, 512), 
		  "conv5_2": (14, 14, 512), 
		  "conv5_3": (14, 14, 512), 
		  "fc6": (4096, ),
		  "fc7": (4096, )
		}

def compute_means_vars_all(): 
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
		# print selected_examples.shape
		selected_examples = selected_examples.reshape((selected_examples.shape[0], -1))
		means[i, :] = np.mean(selected_examples, axis=0)
		covs[i, :, :] = np.cov(selected_examples, rowvar=0)
		vars_[i,:] = np.var(selected_examples, axis=0)

	np.save('means-%s'%(LAYER), means)
	np.save('vars-%s'%(LAYER), vars_)


def compute_mean_vars(num_samples): 
	extracted_feats_path = os.path.join(OUTPUT_PATH, 'feats')

	# Extract first set of labels/activations to get all the dimensions
	with open(os.path.join(extracted_feats_path, 'image_list_%d.dat'%(1)), 'rb') as fp:
		images = cp.load(fp)
	activations = np.load(os.path.join(extracted_feats_path, 'blob-%d.dat.npy'%(1)))

	labels = util.get_attributes('../data/pubfig_attributes.txt', [image.split('/')[-1][:-4] for image in images])
	labels = labels.as_matrix()

	num_attributes = labels.shape[1]
	num_features = np.prod(activations.shape)/activations.shape[0]
	means = np.zeros((num_attributes, num_features))
	# covs = np.zeros((num_attributes, num_features, num_features))
	vars_ = np.zeros((num_attributes, num_features))
	for i in xrange(labels.shape[1]):
		print 'Iteration %d...'%(i+1)
		with open(os.path.join(extracted_feats_path, 'image_list_%d.dat'%(i+1)), 'rb') as fp:
			images = cp.load(fp)
		selected_examples = np.load(os.path.join(extracted_feats_path, 'blob-%d.dat.npy'%(i+1)))
		# print selected_examples.shape
		selected_examples = selected_examples.reshape((selected_examples.shape[0], -1))
		means[i, :] = np.mean(selected_examples, axis=0)
		# covs[i, :, :] = np.cov(selected_examples, rowvar=0)
		vars_[i,:] = np.var(selected_examples, axis=0)

	np.save(os.path.join(OUTPUT_PATH, 'means'), means)
	np.save(os.path.join(OUTPUT_PATH, 'vars'), vars_)

def invert_features(target_feats, layer):
	image_output_path = os.path.join(OUTPUT_PATH, 'outputs')

	L2_REG = 1e-6
	LEARNING_RATE = 1000
	NUM_ITERATIONS = 200
	MAX_JITTER = 4

	solver_path = './DeepFaceNetDeploy.prototxt'
	weights_path = './snapshots/_iter_42000.caffemodel'
	mean_image = np.load("../data/mean_image.npy").astype(np.uint8)

	if not os.path.exists(image_output_path):
		os.makedirs(image_output_path)

	caffe.set_mode_gpu()
	# Load the network
	net = caffe.Net(solver_path, 
					weights_path, 
					caffe.TRAIN)

	# Start with a random image
	# X = np.random.randint(0, 256, size=(224,224,3)).astype(np.float)
	# X -= mean_image
	# X = X[:,:,::-1]

	mean_image_bgr = mean_image[:,:,::-1].astype(np.float)
	# print mean_image_bgr.flatten()[0:50]

	X = np.random.normal(0, 10, (224, 224, 3))
	plt.clf()
	plt.imshow(mean_image)
	plt.axis('off')
	plt.savefig(os.path.join(image_output_path, 'mean-image.png'))
	# out=Image.fromarray(mean_image,mode="RGB")
	# out.save('outputs/mean-image.png')

	# Set up blob data
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))

	data_blob_shape = net.blobs['data'].data.shape
	data_blob_shape = list(data_blob_shape)
	net.blobs['data'].reshape(1, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])
	net.blobs['data'].data[...] = transformer.preprocess('data', X)

	print 'Saving image %d'%(0)
	plt.clf()
	plt.imshow(util.deprocess_image(X, mean_image))
	plt.axis('off')
	plt.savefig(os.path.join(image_output_path, 'image-%d.png'%(0)))

	for t in xrange(1, NUM_ITERATIONS+1):
		# As a regularizer, add random jitter to the image
		ox, oy = np.random.randint(-MAX_JITTER, MAX_JITTER+1, 2)
		X = np.roll(np.roll(X, ox, -1), oy, -2)

		print 'Performing iteration %d...'%(t)
		net.blobs['data'].data[...] = transformer.preprocess('data', X)
		
		feats = net.forward(end=layer)
		net.blobs[layer].diff[...] = 2 * (feats[layer] - target_feats)
		dX = net.backward(start=layer)
		dX = dX['data']
		dX = dX[0, :, :, :]
		dX = np.transpose(dX, (1, 2, 0))

		dX += 2*L2_REG*X
		# print dX.flatten()[0:50]
		X -= LEARNING_RATE*dX

		# Undo the jitter
		X = np.roll(np.roll(X, -ox, -1), -oy, -2)
		
		# As a regularizer, clip the image
		# print X.flatten()[0:50]
		X = np.clip(X, -mean_image_bgr, 255.0 - mean_image_bgr)
		# print X.flatten()[0:50]
		# print '--------------'
		
		# As a regularizer, periodically blur the image
		# if t % blur_every == 0:
		# 	X = blur_image(X)
		
		if t % 10 == 0 or t == NUM_ITERATIONS:
			print 'Saving image %d'%(t)
			plt.clf()
			plt.imshow(util.deprocess_image(X, mean_image))
			plt.axis('off')
			plt.savefig(os.path.join(image_output_path, 'image-%d.png'%(t)))


def main(): 
	# def extract_activations(layer, data_path, weights_path, solver_path, output_path, batch_size=25):
	if len(sys.argv) == 1: 
		mode = 'sample'
	else: 
		mode = sys.argv[1]

	if not os.path.exists(OUTPUT_PATH):
		os.makedirs(OUTPUT_PATH)

	if mode == 'extract':
		print 'Running feature extraction...'
		# util.extract_activations(LAYER, DATA_PATH, WEIGHTS_PATH, SOLVER_PATH, OUTPUT_PATH)
		util.sample_activations(LAYER, DATA_PATH, WEIGHTS_PATH, SOLVER_PATH, OUTPUT_PATH, LAYER_SIZES[LAYER])
		return
	if mode == 'compute':
		print 'Computing mean/vars of extracted feats...' 
		compute_mean_vars(num_samples=NUM_SAMPLES)
		return


	print 'Loading means and vars...'

	means = np.load(os.path.join(OUTPUT_PATH, 'means.npy'))
	vars_ = np.load(os.path.join(OUTPUT_PATH, 'vars.npy'))

	print 'Building GMM...'
	gmm = GMM(means, vars_)

	import time 
	print 'Sampling...'
	target_vec = np.zeros((73,))
	target_vec[0] = 1
	target_vec[57] = 1
	target_vec[9] = 1
	
	target_outs = gmm.sample(target_vec)
	target_outs = target_outs.reshape(LAYER_SIZES[LAYER]).transpose((2,0,1))
	invert_features(target_outs, LAYER)


if __name__ == '__main__':
	main()
