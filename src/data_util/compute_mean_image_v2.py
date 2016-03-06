import os, PIL
from PIL import Image
import numpy as np

# Access all PNG files in directory
base_dir = '../data/eval_set/images_cropped/'
allfiles=os.listdir(base_dir)
imlist=[base_dir + filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]

# Assuming all images are the same size, get dimensions of first image
w,h=(224,224)
N=len(imlist)

# Create a np array of floats to store the average (assume RGB images)
arr=np.zeros((h,w,3),np.float)

# Build up average pixel intensities, casting each image as an array of floats
for idx, im_name in enumerate(imlist):
	print "Image %d/%d"%(idx+1, len(imlist))
	im = Image.open(im_name)
	im = np.array(im.resize((224, 224)), dtype=np.int) # or load whatever ndarray you need
	imarr=np.array(im,dtype=np.float)

	arr=arr+imarr/N

# Round values in array and cast as 8-bit integer
arr=np.array(np.round(arr),dtype=np.uint8)
np.save("mean_eval_image", arr)

# Generate, save and preview final image
out=Image.fromarray(arr,mode="RGB")
out.save("mean_eval_image.png")
out.show()