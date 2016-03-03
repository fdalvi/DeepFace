import numpy as np
import util
import os
from PIL import Image

SET = 'eval'
DATA_PATH = "../data/%s_set/images_cropped"%(SET)

print 'Loading image list....'
images = [os.path.join(DATA_PATH, filename) for filename in util.get_image_names(DATA_PATH, remove_extension=False)]

mean_image = np.zeros((224, 224, 3))
for in_idx, in_ in enumerate(images):
	im = Image.open(in_)
	im = np.array(im.resize((224, 224))) # or load whatever ndarray you need
	mean_image += im

mean_image /= np.float(len(images))

print mean_image.shape
np.save("../data/mean_image", mean_image)