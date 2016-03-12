import numpy as np
import pandas as pd
import os
import random
import cPickle as cp
import caffe

CONSOLIDATED_LABELS = [[0],[1, 2, 3, 57],[4, 5, 6, 7, 8],[9, 10, 11, 58],
						 [12, 28],[13, 14, 15],[16],[17, 18],[19],
						 [20],[21, 22, 23],[24],[25, 26, 27],[29],
						 [30],[31, 32, 33],[34, 35],[36, 37],[38, 39],
						 [40],[41, 42, 43, 44],[45, 46],[47, 48],
						 [49],[50, 51, 52],[53],[54],[55],[56],[59],
						 [60],[61],[62, 63],[64],[65],[66],[67],[68],
						 [69],[70],[71],[72]]

LABELS = ["Male", "Asian", "White", "Black", "Baby", "Child", "Youth", "Middle Aged", "Senior", 
		"Black Hair", "Blond Hair", "Brown Hair", "Bald", "No Eyewear", "Eyeglasses", "Sunglasses", 
		"Mustache", "Smiling", "Frowning", "Chubby", "Blurry", "Harsh Lighting", "Flash", "Soft Lighting", 
		"Outdoor", "Curly Hair", "Wavy Hair", "Straight Hair", "Receding Hairline", "Bangs", "Sideburns", 
		"Fully Visible Forehead", "Partially Visible Forehead", "Obstructed Forehead", "Bushy Eyebrows", 
		"Arched Eyebrows", "Narrow Eyes", "Eyes Open", "Big Nose", "Pointy Nose", "Big Lips", "Mouth Closed", 
		"Mouth Slightly Open", "Mouth Wide Open", "Teeth Not Visible", "No Beard", "Goatee", "Round Jaw", 
		"Double Chin", "Wearing Hat", "Oval Face", "Square Face", "Round Face", "Color Photo", "Posed Photo", 
		"Attractive Man", "Attractive Woman", "Indian", "Gray Hair", "Bags Under Eyes", "Heavy Makeup", "Rosy Cheeks", 
		"Shiny Skin", "Pale Skin", "5 o' Clock Shadow", "Strong Nose-Mouth Lines", "Wearing Lipstick", "Flushed Face", 
		"High Cheekbones", "Brown Eyes", "Wearing Earrings", "Wearing Necktie", "Wearing Necklace"]

'''
	filename: the pubfig_attributes.txt file
	pic_list: 
'''
def get_attributes(filename, pic_list):
	attributes = pd.read_csv(filename, header=1, sep='\t', index_col=(0,1))
	index_list = []
	for im_name in pic_list:
		name_and_num = im_name.split('_')
		num = int(name_and_num[-1])
		name = ' '.join(name_and_num[:-1])
		index_list.append((name, num))

	good_atts = attributes.loc[index_list]
	cols_for_matrix = np.arange(start=2, stop=good_atts.shape[1])
	good_atts.as_matrix(cols_for_matrix)

	good_atts[good_atts > 0] = 1
	good_atts[good_atts <= 0] = 0

	return good_atts

'''
Function to get all the image names in a given directory.

Args:
	path: The path from where the list of images should be obtained.
	remove_extension: boolean defining if the image names should not contain 
		the extension. Default = true

Returns:
	filenames: List of all images in the given path
'''
def get_image_names(path, remove_extension=True): 
	filenames = os.listdir(path)
	if remove_extension:
		filenames = [f.split('.')[0] for f in filenames]
	filenames = [f for f in filenames if len(f) != 0]
	return filenames


'''
Function to consolidate labels. 

Args:
	original_labels: Nx73 matrix 
Returns:
	consolidated_labels: consolidated labels matrix
'''
def consolidate_labels(original_labels, image_names = None, debug=False): 
	original_labels = original_labels.as_matrix()
	consolidated_matrix = np.zeros((original_labels.shape[0], len(CONSOLIDATED_LABELS)), dtype=np.int)

	for i in xrange(original_labels.shape[0]): 
		if image_names is not None:
			if debug: print 'Image ',image_names[i]
		else:
			if debug: print 'Image #', i
		for label_idx, label_list in enumerate(CONSOLIDATED_LABELS): 
			cols = original_labels[:,label_list]

			if np.sum(cols[i,:]) == 0: 
				consolidated_matrix[i,label_idx] = 0
				if debug: print '\tNot ', ",".join([LABELS[l] for l in label_list])
			else: 
				idxs = []
				for idx in xrange(len(label_list)): 
					if cols[i,idx] == 1: 
						idxs.append(idx+1)
				consolidated_matrix[i,label_idx] = random.choice(idxs)
				if debug: print '\t', LABELS[label_list[consolidated_matrix[i,label_idx]-1]]

	return consolidated_matrix


def extract_activations(layer, data_path, weights_path, solver_path, output_path, batch_size=25): 
	assert os.path.exists(data_path)
	assert os.path.exists(weights_path)
	assert os.path.exists(solver_path)
		
	caffe.set_mode_gpu()

	images = get_image_names(data_path)
	num_images = len(images)
	if num_images == 0:
		return

	if not os.path.exists(output_path + 'outs-%s'%(layer)):
		os.makedirs(output_path + 'outs-%s'%(layer))
	with open(output_path + 'outs-%s/image_list.dat'%(layer), 'w+') as fp:
		cp.dump(images, fp)

	images_with_path = [data_path + i + '.jpg' for i in images]
	if num_images % batch_size != 0:
		images_with_path = images_with_path + [images_with_path[-1]]*(batch_size - (num_images % batch_size))

	net = caffe.Net(solver_path,
	                weights_path,
	                caffe.TEST)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))

	data_blob_shape = net.blobs['data'].data.shape
	data_blob_shape = list(data_blob_shape)
	net.blobs['data'].reshape(batch_size, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

	out = np.zeros((num_images, 4096))
	for i in xrange(len(images_with_path) / batch_size):
		start_idx = i*batch_size
		extended_end_idx = (i+1)*batch_size
		end_idx = min(extended_end_idx, num_images)

		print 'Iteration %d of %d'%(i+1, num_images / batch_size + 1)
		net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), \
			images_with_path[i*batch_size:(i+1)*batch_size])
		out[start_idx:end_idx,:] = net.forward(end=layer)[layer][:end_idx-start_idx]

	np.save(output_path + 'outs-%s/blob.dat'%(layer), out)


def sample_activations(layer, data_path, weights_path, solver_path, output_path, layer_dims, batch_size=25, num_samples=100): 
	assert os.path.exists(data_path)
	assert os.path.exists(weights_path)
	assert os.path.exists(solver_path)

	extracted_feats_path = os.path.join(output_path, 'feats')
		
	caffe.set_mode_gpu()

	images = get_image_names(data_path)
	num_images = len(images)
	if num_images == 0:
		return

	if not os.path.exists(extracted_feats_path):
		os.makedirs(extracted_feats_path)
	
	images_with_path = [data_path + i + '.jpg' for i in images]
	if num_images % batch_size != 0:
		images_with_path = images_with_path + [images_with_path[-1]]*(batch_size - (num_images % batch_size))

	##load labels 
	labels = get_attributes('../data/pubfig_attributes.txt', images)
	labels = labels.as_matrix()

	net = caffe.Net(solver_path,
	                weights_path,
	                caffe.TEST)

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

	if 'conv' in layer: 
		layer_dims = (layer_dims[2], layer_dims[0], layer_dims[1])
	transformer.set_transpose('data', (2,0,1))

	data_blob_shape = net.blobs['data'].data.shape
	data_blob_shape = list(data_blob_shape)
	net.blobs['data'].reshape(batch_size, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

	##open activation files and sample images 
	activation_files = []
	out_dim = tuple([num_samples] + list(layer_dims))
	for i in xrange(labels.shape[1]): 
		print 'Label %d of %d'%(i+1, labels.shape[1])
		##sample activations 
		selected_examples = []
		selected_examples_idx = np.random.choice(np.where((labels[:,i] > 0.5))[0], size=100)

		for j in xrange(len(selected_examples_idx)):
			selected_examples.append(images_with_path[selected_examples_idx[j]])
		# print "length of selected examples, attribute %d"%(i), len(selected_examples)

		with open(os.path.join(extracted_feats_path, 'image_list_%d.dat'%(i+1)), 'w+') as fp:
			cp.dump(selected_examples, fp)

		out = np.zeros(out_dim)
		for k in xrange(len(selected_examples) / batch_size):
			start_idx = k*batch_size
			extended_end_idx = (k+1)*batch_size
			end_idx = min(extended_end_idx, num_samples)

			print '\tIteration %d of %d'%(k+1, num_samples / batch_size)
			net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), \
				selected_examples[k*batch_size:(k+1)*batch_size])
			out[start_idx:end_idx,:] = net.forward(end=layer)[layer][:end_idx-start_idx]

		np.save(os.path.join(extracted_feats_path, 'blob-%d.dat'%(i+1)), out)

def extract_batch_activations(layer, data_path, weights_path, solver_path, layer_dims, images, cache=None): 
	assert os.path.exists(data_path)
	assert os.path.exists(weights_path)
	assert os.path.exists(solver_path)
		
	caffe.set_mode_gpu()

	num_images = len(images)
	batch_size = num_images
	if num_images == 0:
		return

	images_with_path = [data_path + i + '.jpg' for i in images]

	##load labels 
	labels = get_attributes('../data/pubfig_attributes.txt', images)
	labels = labels.as_matrix()

	if cache:
		net, transformer = cache
	else:
		net = caffe.Net(solver_path,
	                weights_path,
	                caffe.TEST)

		transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1))
	
	if 'conv' in layer: 
		layer_dims = (layer_dims[2], layer_dims[0], layer_dims[1])
	
	data_blob_shape = net.blobs['data'].data.shape
	data_blob_shape = list(data_blob_shape)
	net.blobs['data'].reshape(batch_size, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])

	out_dim = tuple([batch_size] + list(layer_dims))
	out = np.zeros(out_dim)
	
	net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), \
		images_with_path)
	out = net.forward(end=layer)[layer]

	return labels, out, (net, transformer)

def deprocess_image(X, mean_image):
	r = X.copy()
	r = r.astype(np.uint8)
	r = r[:,:,::-1]
	r += mean_image
	return r


def test():
	##### GET_IMAGE_NAMES test #####
	print get_image_names('../data/dev_set/images_cropped/')

	##### get_attributes test #####
	print get_attributes('../data/pubfig_attributes.txt', ['Aaron_Eckhart_1', 'Aaron_Eckhart_4'])

	pic_list = get_image_names('../data/dev_set/images_cropped/')
	atts = get_attributes('../data/pubfig_attributes.txt', pic_list)
	consolidated_atts = consolidate_labels(atts, pic_list)
	print consolidated_atts
	print consolidated_atts.shape


if __name__ == '__main__':
	test()



