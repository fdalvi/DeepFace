import caffe
import lmdb
from PIL import Image
import os
import numpy as np
import sys
import util

VERBOSE = False
if len(sys.argv) > 1:
	if sys.argv[1] == '--verbose':
		VERBOSE = True

SET = 'eval'
DATA_PATH = "../data/%s_set/images_cropped"%(SET)
ATTR_PATH = "../data/pubfig_attributes.txt"
OUTPUT_PATH = "../data/"

print 'Loading labels....'
images = util.get_image_names(DATA_PATH, remove_extension=True)
labels = util.consolidate_labels(util.get_attributes(ATTR_PATH, images))

print 'Loading image list....'
images = [os.path.join(DATA_PATH, filename) for filename in util.get_image_names(DATA_PATH, remove_extension=False)]

in_db = lmdb.open(os.path.join(OUTPUT_PATH, 'image-%s-lmdb'%(SET)), map_size=int(1e12))
labels_db = lmdb.open(os.path.join(OUTPUT_PATH, 'labels-%s-lmdb'%(SET)), map_size=int(1e12))

in_txn = in_db.begin(write=True)
labels_txn = labels_db.begin(write=True)

for in_idx, in_ in enumerate(images):
	if in_idx % 100 == 0:
		print 'Image %d/%d done.'%(in_idx, len(images))
		# load image:
		# - as np.uint8 {0, ..., 255}
		# - in BGR (switch from RGB)
		# - in Channel x Height x Width order (switch from H x W x C)
		im = np.array(Image.open(in_)) # or load whatever ndarray you need
	if VERBOSE:
		print in_, im.shape
	im = im[:,:,::-1]
	im = im.transpose((2,0,1))
	im_dat = caffe.io.array_to_datum(im)
	in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())

	curr_labels = labels[in_idx, :, None, None]
	labels_dat = caffe.io.array_to_datum(curr_labels)
	labels_txn.put('{:0>10d}'.format(in_idx), labels_dat.SerializeToString())

in_db.close()
labels_db.close()
