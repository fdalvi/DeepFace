import numpy as np
import caffe
import os
import threading
import sys
import util

DATA_PATH = '../data/dev_set/images_cropped/'
WEIGHTS_PATH = 'vgg_face_caffe/VGG_FACE.caffemodel'
SOLVER_PATH = 'DeepFaceNet.prototxt'
BATCH_SIZE = 10

def net_forward(net, transformer, blob, out, iteration, batch_size, num_images, proc_name):
	start_idx = iteration*BATCH_SIZE
	extended_end_idx = (iteration+1)*BATCH_SIZE
	end_idx = min(extended_end_idx, num_images)

	print '[%s] Iteration %d of %d'%(proc_name, iteration+1, num_images / BATCH_SIZE + 1)
	net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), \
		blob)
	# print '[thread=%d]'%iteration, start_idx, end_idx
	out[start_idx:end_idx,:] = net.forward(end='fc7')['fc7'][:end_idx-start_idx]
	# print '[thread=%d]'%iteration, 'done'

def main():
	assert os.path.exists(DATA_PATH)
	assert os.path.exists(WEIGHTS_PATH)
	assert os.path.exists(SOLVER_PATH)

	caffe.set_mode_gpu()

	images = util.get_image_names(DATA_PATH)
	if len(sys.argv) == 1:
		start_idx = 0
		end_idx = len(images)
	else:
		start_idx = int(sys.argv[1])
		end_idx = int(sys.argv[2])
	images = images[start_idx:end_idx]

	if len(images) == 0:
		return

	images_with_path = [DATA_PATH + i + '.jpg' for i in images]
	if len(images) % BATCH_SIZE != 0:
		images_with_path = images_with_path + [images_with_path[-1]]*(BATCH_SIZE - (len(images) % BATCH_SIZE))

	labels = util.get_attributes('../data/pubfig_attributes.txt', images)

	net = caffe.Net(SOLVER_PATH,
	                WEIGHTS_PATH,
	                caffe.TEST)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))

	data_blob_shape = net.blobs['data'].data.shape
	data_blob_shape = list(data_blob_shape)
	net.blobs['data'].reshape(BATCH_SIZE, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

	out_fc7 = np.zeros((len(images), 4096))
	for i in xrange(len(images_with_path) / BATCH_SIZE):
		net_forward(net, 
					transformer, 
					images_with_path[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
					out_fc7,
					i, 
					BATCH_SIZE, 
					len(images),
					'blob_%d_%d'%(start_idx, end_idx))

	if not os.path.exists('outs_fc7'):
		os.makedirs('outs_fc7')
	np.save('outs_fc7/blob_%d_%d.dat'%(start_idx, end_idx), out_fc7)
	print '[blob_%d_%d] done.'%(start_idx, end_idx)

if __name__ == '__main__':
	main()