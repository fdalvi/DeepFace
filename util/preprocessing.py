import sys 
import os 

from utils import *

def preprocessing(): 
	##downloading images
	dev_url_path, eval_url_path = os.path.abspath('../data/dev_set/dev_urls2.txt'), os.path.abspath('../data/eval_set/eval_urls.txt')
	dev_images_patho, eval_images_patho = os.path.abspath('../data/dev_set/dev_images_orig'), os.path.abspath('../data/eval_set/eval_images_orig')
	dev_images_pathc, eval_images_pathc = os.path.abspath('../data/dev_set/dev_images_crop'), os.path.abspath('../data/eval_set/eval_images_crop')

	##create directories if they don't exist
	create_directory(dev_images_patho)
	create_directory(dev_images_pathc)
	create_directory(eval_images_patho)
	create_directory(eval_images_pathc)

	download_images(dev_url_path, dev_images_patho, dev_images_pathc, True)
	download_images(eval_url_path, eval_images_patho, eval_images_pathc, True)

def load_attributes():
	att_path = os.path.abspath('../data/pubfig_attributes.txt')
	get_attributes(att_path, ['Aaron_Eckhart_1', 'Abhishek_Bachan_2'])

if __name__ == '__main__':
	load_attributes()
	# preprocessing()
