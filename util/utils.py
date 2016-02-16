import urllib2
import os 
import httplib
import numpy as np
import pandas as pd

from PIL import Image

URL_POSITION = 3

def create_directory(pathname): 
	if not os.path.exists(pathname): 
		os.makedirs(pathname)

def patch_http_response_read(func):
    def inner(*args):
        try:
            return func(*args)
        except httplib.IncompleteRead, e:
            return e.partial

    return inner
    
httplib.HTTPResponse.read = patch_http_response_read(httplib.HTTPResponse.read)


def request_wrapper(im_info, opener): 
	try: 
		request = urllib2.Request(im_info[URL_POSITION])
		result = opener.open(request)
	except urllib2.HTTPError, e:
		print "image url sucked: " + im_info[0] + '_' + im_info[1] + '_' + im_info[2]
		result = None
	except urllib2.URLError, e:
		print "image url really sucked: " + im_info[0] + '_' + im_info[1] + '_' + im_info[2]
		result = None

	return result


def open_wrapper(result): 
	try: 
		im = Image.open(result)
	except IOError, e: 
		print "bad image"
		im = None

	return im


def download_images(filename, path1, path2, debug=False): 
	f = open(filename, 'r')

	f.readline()
	f.readline()

	for i, line in enumerate(f.readlines()): 
		#download
		im_info = line.split()
		opener = urllib2.build_opener()	
		opener.addheaders = [('User-Agent', 'Mozilla/5.0')]

		result = request_wrapper(im_info, opener)
		if result == None: continue
		# try: 
		# 	request = urllib2.Request(im_info[URL_POSITION])
		# 	result = opener.open(request)
		# except urllib2.HTTPError, e:
		# 	print "image url sucked: " + im_info[0] + '_' + im_info[1] + '_' + im_info[2]
		# 	continue
		# except urllib2.URLError, e:
		# 	print "image url really sucked: " + im_info[0] + '_' + im_info[1] + '_' + im_info[2]
		# 	continue

		if debug: 
			print "im:", im_info

		coords = tuple([int(x) for x in im_info[4].split(',')])
		im = open_wrapper(result)
		if im == None: continue
		# try: 
		# 	im = Image.open(result)
		# except IOError, e: 
		# 	print "bad image"
		# 	continue
		orig_path = os.path.join(path1, im_info[0] + '_' + im_info[1] + '_' + im_info[2] + '.jpg')
		crop_path = os.path.join(path2, im_info[0] + '_' + im_info[1] + '_' + im_info[2] + '_cropped.jpg')
		im.save(orig_path)
		im_cropped = im.crop(coords)
		im_cropped.save(crop_path) 
		
		#break
		# if i == 5: break

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

	



