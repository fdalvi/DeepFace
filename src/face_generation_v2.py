import numpy as np
import os
import random
import caffe
import matplotlib.pyplot as plt
import util
import sklearn
import sys
import cPickle as cp
from PIL import Image

from gmm import GMM

DATA_PATH = '../data/eval_set/images_cropped/'
WEIGHTS_PATH = './snapshots/_iter_42000.caffemodel'
SOLVER_PATH = './DeepFaceNetDeploy.prototxt'
LAYER = 'conv5_1'
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

def conv_forward_naive(x, w, b, conv_param):
	"""
	A naive implementation of the forward pass for a convolutional layer.

	The input consists of N data points, each with C channels, height H and width
	W. We convolve each input with F different filters, where each filter spans
	all C channels and has height HH and width HH.

	Input:
	- x: Input data of shape (N, C, H, W)
	- w: Filter weights of shape (F, C, HH, WW)
	- b: Biases, of shape (F,)
	- conv_param: A dictionary with the following keys:
		- 'stride': The number of pixels between adjacent receptive fields in the
			horizontal and vertical directions.
		- 'pad': The number of pixels that will be used to zero-pad the input.

	Returns a tuple of:
	- out: Output data, of shape (N, F, H', W') where H' and W' are given by
		H' = 1 + (H + 2 * pad - HH) / stride
		W' = 1 + (W + 2 * pad - WW) / stride
	- cache: (x, w, b, conv_param)
	"""
	out = None
	N, C, H, W = x.shape
	F, C, HH, WW = w.shape
	pad = conv_param['pad']
	stride = conv_param['stride']
	
	x_padded = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
	H_out = 1 + (H + 2 * pad - HH) / stride
	W_out = 1 + (W + 2 * pad - WW) / stride
	out = np.zeros((N, F, H_out, W_out))

	for i in xrange(H_out):
		for j in xrange(W_out):
			start_h = i * stride
			end_h = start_h + HH
			start_w = j * stride
			end_w = start_w + WW
			
			out[:, :, i,j] = np.dot(x_padded[:, :, start_h:end_h, start_w:end_w].reshape((N, -1)), w.reshape((F, -1)).T) + b
	cache = (x, w, b, conv_param)
	return out, cache

def blur_image(X):
  """
  A very gentle image blurring operation, to be used as a regularizer for image
  generation.
  
  Inputs:
  - X: Image data of shape (N, 3, H, W)
  
  Returns:
  - X_blur: Blurred version of X, of shape (N, 3, H, W)
  """
  w_blur = np.zeros((3, 3, 3, 3))
  b_blur = np.zeros(3)
  blur_param = {'stride': 1, 'pad': 1}
  for i in xrange(3):
    w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
  w_blur /= 200.0
  return conv_forward_naive(np.expand_dims(X, 0), w_blur, b_blur, blur_param)[0][0, :, :, :]

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

def invert_features(target_feats, layer, target_image = None, blur_every = 1):
	image_output_path = os.path.join(OUTPUT_PATH, 'outputs')

	L2_REG = 1e-6
	LEARNING_RATE = 1e-2
	NUM_ITERATIONS = 500
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
	mean_image_bgr_T = np.transpose(mean_image_bgr, (2, 0, 1))
	# print mean_image_bgr.flatten()[0:50]

	# net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images_with_path)

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

	if target_image:
		X = np.array(Image.open(target_image))
		X -= mean_image
		X = X[:,:,::-1]
		net.blobs['data'].data[...] = transformer.preprocess('data', X)

		print 'Saving target image'
		plt.clf()
		plt.imshow(util.deprocess_image(X, mean_image))
		plt.axis('off')
		plt.savefig(os.path.join(image_output_path, 'target_image.png'))
		target_feats_input = target_feats.copy()
		target_feats = net.forward(end=layer)[layer].copy()
		# print np.mean(target_feats/target_feats_input)

		# barack - sampled = -1e4
		# barack - 1000*sampled = 1e4
	# return	

	X = np.random.normal(0, 255, (224, 224, 3))
	net.blobs['data'].data[...] = transformer.preprocess('data',X)
	X = np.transpose(X, (2, 0, 1))

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
		X = np.transpose(X, (1, 2, 0))
		net.blobs['data'].data[...] = transformer.preprocess('data', X)
		X = np.transpose(X, (2, 0 ,1))

		feats = net.forward(end=layer)
		print '\t Difference: %f'%(np.sum(np.abs(feats[layer] - target_feats)))
		net.blobs[layer].diff[...] = 2 * (feats[layer] - target_feats)
		dX = net.backward(start=layer)
		dX = dX['data']
		dX = dX[0, :, :, :]
		# dX = np.transpose(dX, (1, 2, 0))

		dX += 2*L2_REG*X
		# print dX.flatten()[0:50]
		X -= LEARNING_RATE*dX

		# Undo the jitter
		X = np.roll(np.roll(X, -ox, -1), -oy, -2)
		
		# As a regularizer, clip the image
		# print X.flatten()[0:50]
		X = np.clip(X, -mean_image_bgr_T, 255.0 - mean_image_bgr_T)
		# print X.flatten()[0:50]
		# print '--------------'
		
		# As a regularizer, periodically blur the image
		if t % blur_every == 0:
			X = blur_image(X)
		
		if t % 10 == 0 or t == NUM_ITERATIONS:
			print 'Saving image %d'%(t)
			plt.clf()
			plt.imshow(util.deprocess_image(X, mean_image))
			plt.axis('off')
			plt.savefig(os.path.join(image_output_path, 'image-%d.png'%(t)))


def main(): 
		# 0: "Male"
		# 1: "Asian"
		# 2: "White"
		# 3: "Black"
		# 4: "Baby"
		# 5: "Child"
		# 6: "Youth"
		# 7: "Middle Aged"
		# 8: "Senior"
		# 9: "Black Hair"
		# 10: "Blond Hair"
		# 11: "Brown Hair"
		# 12: "Bald"
		# 13: "No Eyewear"
		# 14: "Eyeglasses"
		# 15: "Sunglasses"
		# 16: "Mustache"
		# 17: "Smiling"
		# 18: "Frowning"
		# 19: "Chubby"
		# 20: "Blurry"
		# 21: "Harsh Lighting"
		# 22: "Flash"
		# 23: "Soft Lighting"
		# 24: "Outdoor"
		# 25: "Curly Hair"
		# 26: "Wavy Hair"
		# 27: "Straight Hair"
		# 28: "Receding Hairline"
		# 29: "Bangs"
		# 30: "Sideburns"
		# 31: "Fully Visible Forehead"
		# 32: "Partially Visible Forehead"
		# 33: "Obstructed Forehead"
		# 34: "Bushy Eyebrows"
		# 35: "Arched Eyebrows"
		# 36: "Narrow Eyes"
		# 37: "Eyes Open"
		# 38: "Big Nose"
		# 39: "Pointy Nose"
		# 40: "Big Lips"
		# 41: "Mouth Closed"
		# 42: "Mouth Slightly Open"
		# 43: "Mouth Wide Open"
		# 44: "Teeth Not Visible"
		# 45: "No Beard"
		# 46: "Goatee"
		# 47: "Round Jaw"
		# 48: "Double Chin"
		# 49: "Wearing Hat"
		# 50: "Oval Face"
		# 51: "Square Face"
		# 52: "Round Face"
		# 53: "Color Photo"
		# 54: "Posed Photo"
		# 55: "Attractive Man"
		# 56: "Attractive Woman"
		# 57: "Indian"
		# 58: "Gray Hair"
		# 59: "Bags Under Eyes"
		# 60: "Heavy Makeup"
		# 61: "Rosy Cheeks"
		# 62: "Shiny Skin"
		# 63: "Pale Skin"
		# 64: "5 o' Clock Shadow"
		# 65: "Strong Nose-Mouth Lines"
		# 66: "Wearing Lipstick"
		# 67: "Flushed Face"
		# 68: "High Cheekbones"
		# 69: "Brown Eyes"
		# 70: "Wearing Earrings"
		# 71: "Wearing Necktie"
		# 72: "Wearing Necklace"
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
	if mode == 'train':
		print 'Loading means and vars...'

		means = np.load(os.path.join(OUTPUT_PATH, 'means.npy'))
		vars_ = np.load(os.path.join(OUTPUT_PATH, 'vars.npy'))

		print 'Building GMM...'
		g = GMM(means, vars_)

		g.train(LAYER, DATA_PATH, WEIGHTS_PATH, SOLVER_PATH, OUTPUT_PATH, LAYER_SIZES[LAYER], num_iterations=1000, batch_size=25, save_every=50)
		return

	print 'Loading means and vars...'
	means = np.load(os.path.join(OUTPUT_PATH, 'means.npy'))
	vars_ = np.load(os.path.join(OUTPUT_PATH, 'vars.npy'))

	print 'Building GMM...'
	gmm = GMM(means, vars_)
	weights_output_path = os.path.join(OUTPUT_PATH, 'weights')
	gmm.load_weights(os.path.join(weights_output_path, 'weights_iter250.npy'))

	print 'Sampling...'
	target_vec = np.zeros((73,))
	# target_vec[0] = 1 # Male
	# target_vec[3] = 1 # black
	# target_vec[7] = 1 # middle aged
	# target_vec[9] = 1 # black hair
	# target_vec[13] = 1 # no eyewear
	# target_vec[17] = 1 # smiling
	# target_vec[31] = 1 # visible forehead
	# target_vec[37] = 1 # open eyes
	# target_vec[53] = 1 # color photo
	# target_vec[69] = 1 # brown eyes
	target_vec[11] = 1 # brown hair
	target_vec[18] = 1 # frowning

	target_outs = gmm.sample(target_vec)
	target_outs = target_outs.reshape(LAYER_SIZES[LAYER]).transpose((2,0,1))
	# invert_features(target_outs, LAYER, '../data/dev_set/images_cropped/Jared_Leto_116.jpg')
	# invert_features(target_outs, LAYER, '../data/dev_set/images_cropped/Barack_Obama_153.jpg')
	# invert_features(target_outs, LAYER, '../data/eval_set/images_cropped/Adriana_Lima_239.jpg')
	invert_features(target_outs, LAYER)


if __name__ == '__main__':
	main()
