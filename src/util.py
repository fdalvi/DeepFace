import numpy as np
import pandas as pd
import os

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
	# print good_atts
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
	return filenames


def test():
	print get_image_names('../data/dev_set/images_cropped/')


if __name__ == '__main__':
	test()



