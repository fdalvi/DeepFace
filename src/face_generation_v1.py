import numpy as np
import os
import random
import caffe


def class_visualization(target_y): 
	solver_path = './DeepFaceNetDeploy.prototxt'
	weights_path = './snapshots/_iter_20000.caffemodel'

	net = caffe.Net(solver_path, 
					weights_path, 
					caffe.TEST)


	X = np.random.randn(224,224,3)

	##set up blob data
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))

	data_blob_shape = net.blobs['data'].data.shape
	data_blob_shape = list(data_blob_shape)
	net.blobs['data'].reshape(1, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])
	net.blobs['data'].data[...] = transformer.preprocess('data', X)

	all_diffs = []
	for i in xrange(1,43):
		diff = net.blobs['fc8-%d'%(i)].diff
		one_hot = np.zeros_like(diff)
		one_hot[0, target_y[i-1]] = 1
		net.blobs['fc8-%d'%(i)].data[...] = one_hot
		net.blobs['fc8-%d'%(i)].diff[...] = np.zeros_like(diff)
		print net.blobs['fc8-%d'%(i)].data
		# all_diffs.append(net.blobs['fc8-%d'%(i)])
		all_diffs.append('fc8-%d'%(i))


	# print net.blobs['data'].data
	# print type(net.blobs['data'].data)
	# print net.blobs['data'].data.shape
	_ = net.forward()
	dX = net.backward()

	# X, dX = net.forward_backward_all(blobs=[net.blobs['data']], diffs=all_diffs, None)
	# X, dX = net.forward_backward_all()
	print dX
	print net.blobs['conv1_1'].diff
	print net.blobs['data'].diff


	# X = np.random.randn(1, 3, 64, 64)
	# scores, _ = model.forward(X)
	# one_hot_scores = np.zeros(scores.shape)
	# one_hot_scores[0,target_y] = 1
	# for t in xrange(num_iterations):
	# 	# As a regularizer, add random jitter to the image
	# 	ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
	# 	X = np.roll(np.roll(X, ox, -1), oy, -2)

	# 	dX = None
	# 	############################################################################
	# 	# TODO: Compute the image gradient dX of the image with respect to the     #
	# 	# target_y class score. This should be similar to the fooling images. Also #
	# 	# add L2 regularization to dX and update the image X using the image       #
	# 	# gradient and the learning rate.                                          #
	# 	############################################################################
	# 	_, cache = model.forward(X, mode="test")
	# 	dX, grads = model.backward(one_hot_scores, cache)
	# 	dX -= 2*l2_reg*X
	# 	X += learning_rate*dX

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
