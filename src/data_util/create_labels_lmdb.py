import caffe
import lmdb
import numpy as np
from PIL import Image

labels = np.array([[0,1], [1,0], [0,0]])

in_db = lmdb.open('labels-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(labels):
        im = in_[:, None, None]
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()