import matplotlib.pyplot as plt
import numpy as np
import os
import util

DATA_PATH = '../data/eval_set/images_cropped/'
WEIGHTS_PATH = './snapshots/_iter_42000.caffemodel'
SOLVER_PATH = './DeepFaceNetDeploy.prototxt'
LABELS = ["Male",
			"Asian",
			"White",
			"Black",
			"Baby",
			"Child",
			"Youth",
			"Middle Aged",
			"Senior",
			"Black Hair",
			"Blond Hair",
			"Brown Hair",
			"Bald",
			"No Eyewear",
			"Eyeglasses",
			"Sunglasses",
			"Mustache",
			"Smiling",
			"Frowning",
			"Chubby",
			"Blurry",
			"Harsh Lighting",
			"Flash",
			"Soft Lighting",
			"Outdoor",
			"Curly Hair",
			"Wavy Hair",
			"Straight Hair",
			"Receding Hairline",
			"Bangs",
			"Sideburns",
			"Full Forehead",
			"Partial Forehead",
			"Obstructed Forehead",
			"Bushy Eyebrows",
			"Arched Eyebrows",
			"Narrow Eyes",
			"Eyes Open",
			"Big Nose",
			"Pointy Nose",
			"Big Lips",
			"Mouth Closed",
			"Mouth Slightly Open",
			"Mouth Wide Open",
			"Teeth Not Visible",
			"No Beard",
			"Goatee",
			"Round Jaw",
			"Double Chin",
			"Wearing Hat",
			"Oval Face",
			"Square Face",
			"Round Face",
			"Color Photo",
			"Posed Photo",
			"Attractive Man",
			"Attractive Woman",
			"Indian",
			"Gray Hair",
			"Bags Under Eyes",
			"Heavy Makeup",
			"Rosy Cheeks",
			"Shiny Skin",
			"Pale Skin",
			"5 o' Clock Shadow",
			"Nose-Mouth Lines",
			"Wearing Lipstick",
			"Flushed Face",
			"High Cheekbones",
			"Brown Eyes",
			"Wearing Earrings",
			"Wearing Necktie",
			"Wearing Necklace"]


def main():
	assert os.path.exists(DATA_PATH)
	assert os.path.exists(WEIGHTS_PATH)
	assert os.path.exists(SOLVER_PATH)

	images = util.get_image_names(DATA_PATH)
	num_images = len(images)

	images_with_path = [DATA_PATH + i + '.jpg' for i in images]

	##load labels 
	labels = util.get_attributes('../data/pubfig_attributes.txt', images)
	labels = labels.as_matrix()

	fig, ax = plt.subplots()

	x = []
	y = []
	for i in xrange(labels.shape[1]): 
		print 'Label %d of %d'%(i+1, labels.shape[1])
		x.append(i)
		y.append(np.sum((labels[:,i] > 0.5)))

	print len(x), len(y)
	plt.bar(x, y, color="blue")
	# ax.set_xlim([0.0,20.0])
	
	ax.set_xticks(range(len(LABELS)))
	ax.set_xticklabels(LABELS, rotation=90, size=8)
	# ax.get_xaxis().majorTicks[2].set_pad(50)
	plt.gcf().subplots_adjust(bottom=0.30)
	plt.show()
	plt.savefig('distrib.png')

if __name__ == '__main__':
	main()