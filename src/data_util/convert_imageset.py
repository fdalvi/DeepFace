import caffe
import lmdb
from PIL import Image
import os
import numpy as np
import sys
import util

VERBOSE = False
SET = 'dev'
if len(sys.argv) > 1:
	if sys.argv[1] == '--verbose':
		VERBOSE = True
	if sys.argv[1] == 'eval':
		SET = 'eval'
	if sys.argv[1] == 'dev':
		SET = 'dev'

DATA_PATH = "../data/%s_set/images_cropped"%(SET)
ATTR_PATH = "../data/pubfig_attributes.txt"
OUTPUT_PATH = "../data/"

print 'Loading mean image...'
mean_image = np.load("../data/mean_image.npy").astype(np.int)

print 'Loading labels....'
images = util.get_image_names(DATA_PATH, remove_extension=True)
labels = util.consolidate_labels(util.get_attributes(ATTR_PATH, images))

print 'Loading image list....'
images = [os.path.join(DATA_PATH, filename) for filename in util.get_image_names(DATA_PATH, remove_extension=False)]

in_db = lmdb.open(os.path.join(OUTPUT_PATH, 'image-%s-lmdb'%(SET)), map_size=int(1e12))
labels_db = lmdb.open(os.path.join(OUTPUT_PATH, 'labels-%s-lmdb'%(SET)), map_size=int(1e12))

with in_db.begin(write=True) as in_txn:
	for in_idx, in_ in enumerate(images):
		if in_idx % 100 == 0:
			print 'Image %d/%d done.'%(in_idx, len(images))

		# load image:
		# - as np.uint8 {0, ..., 255}
		# - in BGR (switch from RGB)
		# - in Channel x Height x Width order (switch from H x W x C)
		im = Image.open(in_)
		im = np.array(im.resize((224, 224)), dtype=np.int) # or load whatever ndarray you need
		im -= mean_image

		if VERBOSE:
			print in_, im.shape
		im = im[:,:,::-1]
		im = im.transpose((2,0,1))
		im_dat = caffe.io.array_to_datum(im)
		in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

with labels_db.begin(write=True) as labels_txn:
	for in_idx, in_ in enumerate(images):
		if in_idx % 100 == 0:
			print 'Label %d/%d done.'%(in_idx, len(images))
		curr_labels = labels[in_idx, :, None, None]
		labels_dat = caffe.io.array_to_datum(curr_labels)
		labels_txn.put('{:0>10d}'.format(in_idx), labels_dat.SerializeToString())
labels_db.close()
