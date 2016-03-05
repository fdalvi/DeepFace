import numpy as np
import os
import random
import caffe
import matplotlib.pyplot as plt

def deprocess_image(X, mean_image):
	r = X.copy()
	r = r.astype(np.int)
	r += mean_image
	return r

def class_visualization(target_y): 
	L2_REG = 1e-6
	LEARNING_RATE = 20000
	NUM_ITERATIONS = 200
	MAX_JITTER = 4

	solver_path = './DeepFaceNetDeploy.prototxt'
	weights_path = './snapshots/_iter_20000.caffemodel'
	mean_image = np.load("../data/mean_image.npy").astype(np.int)

	# Load the network
	net = caffe.Net(solver_path, 
					weights_path, 
					caffe.TRAIN)

	# Start with a random image
	X = np.random.randint(0, 256, size=(224,224,3)).astype(np.float)
	X -= mean_image

	# Set up blob data
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))

	data_blob_shape = net.blobs['data'].data.shape
	data_blob_shape = list(data_blob_shape)
	net.blobs['data'].reshape(1, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])
	net.blobs['data'].data[...] = transformer.preprocess('data', X)

	# Set the target diffs at fc8 layers
	one_hots = []
	for i in xrange(1,43):
		diff = net.blobs['fc8-%d'%(i)].diff
		one_hot = np.zeros_like(diff)
		one_hot[0, target_y[i-1]] = 1

		net.blobs['fc8-%d'%(i)].diff[...] = one_hot

		one_hots.append(one_hot)

	# print 'Before'
	# print net.blobs['fc8-1'].diff
	# print net.blobs['fc8-42'].diff
	# _ = net.forward()
	# dX = net.backward(start='fc8-42')
	# print 'After'
	# print net.blobs['fc8-1'].diff
	# print net.blobs['fc8-42'].diff

	# print mean_image.flatten()[0:10]
	for t in xrange(NUM_ITERATIONS):
		# As a regularizer, add random jitter to the image
		ox, oy = np.random.randint(-MAX_JITTER, MAX_JITTER+1, 2)
		X = np.roll(np.roll(X, ox, -1), oy, -2)

		print 'Performing iteration %d...'%(t+1)
		net.blobs['data'].data[...] = transformer.preprocess('data', X)
		for i in xrange(1,43):
			net.blobs['fc8-%d'%(i)].diff[...] = one_hots[i-1]

		_ = net.forward()
		dX = net.backward(start='fc8-42')
		dX = dX['data']
		dX = dX[0, :, :, :]
		dX = np.transpose(dX, (1, 2, 0))

		dX -= 2*L2_REG*X
		X += LEARNING_RATE*dX

		# Undo the jitter
		X = np.roll(np.roll(X, -ox, -1), -oy, -2)
		
		# As a regularizer, clip the image
		X = np.clip(X, -mean_image, 255.0 - mean_image)
		
		# As a regularizer, periodically blur the image
		# if t % blur_every == 0:
		# 	X = blur_image(X)
		
		if t % 10 == 0:
			print 'Saving image %d'%(t)
			plt.clf()
			plt.imshow(deprocess_image(X, mean_image))
			plt.axis('off')
			plt.savefig('outputs/image-%d.png'%(t))

def main(): 
	target_y = np.array([0]*42)
	target_y[5] = 1
	target_y[10] = 1
	class_visualization(target_y)


if __name__ == '__main__':
	main()
