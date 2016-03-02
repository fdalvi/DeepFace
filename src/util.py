import numpy as np
import pandas as pd
import os
import random

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
	return filenames


'''
Function to consolidate labels. 

Args:
	original_labels: Nx73 matrix 
Returns:
	consolidated_labels: consolidated labels matrix
'''
def consolidate_labels(original_labels): 
	original_labels = original_labels.as_matrix()
	print type(original_labels)
	consolidated_matrix = np.zeros((original_labels.shape[0], len(CONSOLIDATED_LABELS)))

	for label_idx, label_list in enumerate(CONSOLIDATED_LABELS): 
		cols = original_labels[:,label_list]
		for i in xrange(cols.shape[0]): 
			if np.sum(cols[i,:]) == 0: 
				consolidated_matrix[i,label_idx] = 0
			else: 
				idxs = []
				for idx in xrange(len(label_list)): 
					if cols[i,idx] == 1: 
						idxs.append(idx+1)
				consolidated_matrix[i,label_idx] = random.choice(idxs)

	print original_labels[3,:]
	print consolidated_matrix[3,:]
	for i in xrange(original_labels.shape[1]): 
		if original_labels[3,i] == 0: 
			print "not " + LABELS[i]
		else: 
			print LABELS[i]

	for i in xrange(consolidated_matrix.shape[1]): 
		print "column: %d, consolidated label: %d" % (i, consolidated_matrix[3,i])
	return None


def test():
	pic_list = get_image_names('../data/dev_set/images_cropped/')
	atts = get_attributes('../data/pubfig_attributes.txt', pic_list)
	print pic_list[3]
	print consolidate_labels(atts)


if __name__ == '__main__':
	test()



