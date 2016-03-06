import numpy as np
import os
import random
import caffe
import matplotlib.pyplot as plt

def deprocess_image(X, mean_image):
	r = X.copy()
	r = r.astype(np.uint8)
	r = r[:,:,::-1]
	r += mean_image
	return r

def class_visualization(target_y): 
	L2_REG = 1e-6
	LEARNING_RATE = 20000
	NUM_ITERATIONS = 200
	MAX_JITTER = 4

	solver_path = './DeepFaceNetDeploy.prototxt'
	weights_path = './snapshots/_iter_42000.caffemodel'
	mean_image = np.load("../data/mean_image.npy").astype(np.uint8)

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
	plt.savefig('outputs/mean-image.png')
	# out=Image.fromarray(mean_image,mode="RGB")
	# out.save('outputs/mean-image.png')

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
	
	print 'Saving image %d'%(0)
	plt.clf()
	plt.imshow(deprocess_image(X, mean_image))
	plt.axis('off')
	plt.savefig('outputs/image-%d.png'%(t))

	# print mean_image.flatten()[0:10]
	for t in xrange(1, NUM_ITERATIONS+1):
		# As a regularizer, add random jitter to the image
		ox, oy = np.random.randint(-MAX_JITTER, MAX_JITTER+1, 2)
		X = np.roll(np.roll(X, ox, -1), oy, -2)

		print 'Performing iteration %d...'%(t)
		net.blobs['data'].data[...] = transformer.preprocess('data', X)
		for i in xrange(1,43):
			net.blobs['fc8-%d'%(i)].diff[...] = one_hots[i-1]

		_ = net.forward()
		dX = net.backward(start='fc8-42')
		dX = dX['data']
		dX = dX[0, :, :, :]
		dX = np.transpose(dX, (1, 2, 0))

		dX -= 2*L2_REG*X
		# print dX.flatten()[0:50]
		X += LEARNING_RATE*dX

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
			plt.imshow(deprocess_image(X, mean_image))
			plt.axis('off')
			plt.savefig('outputs/image-%d.png'%(t))

def main(): 
	# 0: "Male",
	# 1: "Asian", "White", "Black", "Indian",
	# 2: "Baby", "Child", "Youth", "Middle Aged", "Senior",
	# 3: "Black Hair", "Blond Hair", "Brown Hair", "Gray Hair",
	# 4: "Bald", "Receding Hairline",
	# 5: "No Eyewear", "Eyeglasses", "Sunglasses",
	# 6: "Mustache",
	# 7: "Smiling", "Frowning",
	# 8: "Chubby",
	# 9: "Blurry",
	# 10: "Harsh Lighting", "Flash", "Soft Lighting",
	# 11: "Outdoor",
	# 12: "Curly Hair", "Wavy Hair", "Straight Hair",
	# 13: "Bangs",
	# 14: "Sideburns",
	# 15: "Fully Visible Forehead", "Partially Visible Forehead", "Obstructed Forehead",
	# 16: "Bushy Eyebrows", "Arched Eyebrows",
	# 17: "Narrow Eyes", "Eyes Open",
	# 18: "Big Nose", "Pointy Nose",
	# 19: "Big Lips",
	# 20: "Mouth Closed", "Mouth Slightly Open", "Mouth Wide Open", "Teeth Not Visible",
	# 21: "No Beard", "Goatee",
	# 22: "Round Jaw", "Double Chin",
	# 23: "Wearing Hat",
	# 24: "Oval Face", "Square Face", "Round Face",
	# 25: "Color Photo",
	# 26: "Posed Photo",
	# 27: "Attractive Man",
	# 28: "Attractive Woman",
	# 29: "Bags Under Eyes",
	# 30: "Heavy Makeup",
	# 31: "Rosy Cheeks",
	# 32: "Shiny Skin", "Pale Skin",
	# 33: "5 o' Clock Shadow",
	# 34: "Strong Nose-Mouth Lines",
	# 35: "Wearing Lipstick",
	# 36: "Flushed Face",
	# 37: "High Cheekbones",
	# 38: "Brown Eyes",
	# 39: "Wearing Earrings",
	# 40: "Wearing Necktie",
	# 41: "Wearing Necklace",
	target_y = np.array([0]*42)
	target_y[0] = 1 # Male
	target_y[1] = 3 # Black
	target_y[2] = 3 # Youth
	target_y[3] = 1 # Black hair
	class_visualization(target_y)


if __name__ == '__main__':
	main()
