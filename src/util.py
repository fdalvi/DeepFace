import numpy as np
import pandas as pd

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
