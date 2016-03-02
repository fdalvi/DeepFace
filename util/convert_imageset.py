import caffe
import lmdb
from PIL import Image
import os
import numpy as np
import sys

VERBOSE = False
if len(sys.argv) > 1:
	if sys.argv[1] == '--verbose':
		VERBOSE = True


DATA_PATH = "../data/eval_set/images_cropped"
inputs = [os.path.join(DATA_PATH,f) for f in os.listdir(DATA_PATH)]
#print inputs

in_db = lmdb.open('image-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	for in_idx, in_ in enumerate(inputs):
		if in_idx % 100 == 0:
			print 'Image %d/%d done.'%(in_idx, len(inputs))
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
in_db.close()
