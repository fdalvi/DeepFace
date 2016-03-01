import numpy as np
import caffe
import os
import threading
import sys
import util
import cPickle as cp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

DATA_PATH = '../data/dev_set/images_cropped/'
WEIGHTS_PATH = 'vgg_face_caffe/VGG_FACE.caffemodel'
SOLVER_PATH = 'vgg_face_caffe/VGG_FACE_deploy.prototxt'
BATCH_SIZE = 10
OUTPUT_PATH = 'analysis/'

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

def plot_tsne(fc7, y):
	print 'Performing TSNE...'
	visualizer = TSNE(n_components=2, random_state=0)
	fc7_r = visualizer.fit_transform(fc7) 

	print 'Saving plots...'
	if not os.path.exists(OUTPUT_PATH + 'feature_maps'):
			os.makedirs(OUTPUT_PATH + 'feature_maps')
	for i in xrange(y.shape[1]):
		plt.scatter(fc7_r[:,0], fc7_r[:, 1], c=plt.cm.nipy_spectral(y[:, i]/2.0 + 0.45))
		plt.savefig(OUTPUT_PATH + 'feature_maps/' + str(i)+'.png')
		plt.clf()
		print 'Saved plot %d'%(i+1)
    # handles = [mpatches.Patch(color=plt.cm.rainbow(1.0*i/9.0), label='Class ' + str(i)) for i in xrange(0,10)]
    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.show()

def main():
	if len(sys.argv) == 1:
		mode = "PLOT"
	else:
		if sys.argv[1] == "PLOT" or sys.argv[1] == "EXTRACT":
			mode = sys.argv[1]
		else:
			print 'Usage: python vgg_visualizer.py EXTRACT|PLOT'
			sys.exit(1)

	if mode == "EXTRACT":
		assert os.path.exists(DATA_PATH)
		assert os.path.exists(WEIGHTS_PATH)
		assert os.path.exists(SOLVER_PATH)

		caffe.set_mode_gpu()

		images = util.get_image_names(DATA_PATH)

		start_idx = 0
		end_idx = len(images)
		# TODO: Use command line arguments to split into batches if necessary
		# start_idx = int(sys.argv[1])
		# end_idx = int(sys.argv[2])
		images = images[start_idx:end_idx]

		if len(images) == 0:
			return

		if not os.path.exists(OUTPUT_PATH + 'outs_fc7'):
			os.makedirs(OUTPUT_PATH + 'outs_fc7')
		with open(OUTPUT_PATH + 'outs_fc7/image_list.dat', 'w+') as fp:
			cp.dump(images, fp)

		images_with_path = [DATA_PATH + i + '.jpg' for i in images]
		if len(images) % BATCH_SIZE != 0:
			images_with_path = images_with_path + [images_with_path[-1]]*(BATCH_SIZE - (len(images) % BATCH_SIZE))

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

		np.save(OUTPUT_PATH + 'outs_fc7/blob_%d_%d.dat'%(start_idx, end_idx), out_fc7)
		print '[blob_%d_%d] done.'%(start_idx, end_idx)
	else:
		# Visualize
		with open(OUTPUT_PATH + 'outs_fc7/image_list.dat', 'rb') as fp:
			images = cp.load(fp)

		start_idx = 0
		end_idx = len(images)

		# TODO: Use command line arguments to split into batches if necessary
		# start_idx = int(sys.argv[1])
		# end_idx = int(sys.argv[2])

		out_fc7 = np.load(OUTPUT_PATH + 'outs_fc7/blob_%d_%d.dat.npy'%(start_idx, end_idx))
		labels = util.get_attributes('../data/pubfig_attributes.txt', images)
		print labels.shape
		print out_fc7.shape

		labels = labels.as_matrix()

		plot_tsne(out_fc7, labels)

if __name__ == '__main__':
	main()