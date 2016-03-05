import numpy as np
import os
import random
import caffe
import matplotlib.pyplot as plt

def class_visualization(target_y): 
	L2_REG = 1e-6
	learning_rate = 10000
	solver_path = './DeepFaceNetDeploy.prototxt'
	weights_path = './snapshots/_iter_20000.caffemodel'

	# Load the network
	net = caffe.Net(solver_path, 
					weights_path, 
					caffe.TRAIN)

	# Start with a random image
	X = np.random.randn(224,224,3)

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
		# net.blobs['fc8-%d'%(i)].data[...] = one_hot
		net.blobs['fc8-%d'%(i)].diff[...] = one_hot
		one_hots.append(one_hot)

	for t in xrange(10):
	# 	# As a regularizer, add random jitter to the image
	# 	ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
	# 	X = np.roll(np.roll(X, ox, -1), oy, -2)
		print 'Performing forward pass...'
		net.blobs['data'].data[...] = transformer.preprocess('data', X)
		for i in xrange(1,43):
			net.blobs['fc8-%d'%(i)].diff[...] = one_hots[i-1]

		_ = net.forward()
		dX = net.backward()
		dX = dX['data']
		dX = dX[0, :, :, :]
		dX = np.transpose(dX, (1, 2, 0))

		dX -= 2*L2_REG*X
		X += learning_rate*dX

		print 'done...'
		# deprocess_image(X, data['mean_image'])
		plt.imshow(X)
		# plt.gcf().set_size_inches(3, 3)
		plt.axis('off')
		plt.savefig('outputs/image-%d.png'%(t))
 #  
	# 	############################################################################
	# 	#                             END OF YOUR CODE                             #
	# 	############################################################################
		
	# 	# Undo the jitter
	# 	X = np.roll(np.roll(X, -ox, -1), -oy, -2)
		
	# 	# As a regularizer, clip the image
	# 	X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])
		
	# 	# As a regularizer, periodically blur the image
	# 	if t % blur_every == 0:
	# 		X = blur_image(X)
		
	# 	# Periodically show the image
	# 	if t % show_every == 0:
	# 		plt.imshow(deprocess_image(X, data['mean_image']))
	# 		plt.gcf().set_size_inches(3, 3)
	# 		plt.axis('off')
	# 		plt.show()
 #  return X

def main(): 
	target_y = np.array([0]*42)
	target_y[5] = 1
	target_y[10] = 1
	class_visualization(target_y)


if __name__ == '__main__':
	main()
