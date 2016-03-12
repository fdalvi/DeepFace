import gmm
import numpy as np
import os

DATA_PATH = '../data/eval_set/images_cropped/'
WEIGHTS_PATH = './snapshots/_iter_42000.caffemodel'
SOLVER_PATH = './DeepFaceNetDeploy.prototxt'
LAYER = 'conv3_1'
NUM_SAMPLES = 100
OUTPUT_PATH = './trial_%s_%d/'%(LAYER, NUM_SAMPLES)

LAYER_SIZES = {
		  "conv1_1": (224, 224, 64), 
		  "conv1_2": (224, 224, 64), 
		  "conv2_1": (112, 112, 128), 
		  "conv2_2": (112, 112, 128), 
		  "conv3_1": (56, 56, 256), 
		  "conv3_2": (56, 56, 256), 
		  "conv3_3": (56, 56, 256), 
		  "conv4_1": (28, 28, 512), 
		  "conv4_2": (28, 28, 512), 
		  "conv4_3": (28, 28, 512), 
		  "conv5_1": (14, 14, 512), 
		  "conv5_2": (14, 14, 512), 
		  "conv5_3": (14, 14, 512), 
		  "fc6": (4096, ),
		  "fc7": (4096, )
		}

def main():
	print 'Loading means and vars...'

	means = np.load(os.path.join(OUTPUT_PATH, 'means.npy'))
	vars_ = np.load(os.path.join(OUTPUT_PATH, 'vars.npy'))

	print 'Building GMM...'
	g = gmm.GMM(means, vars_)

	g.train(LAYER, DATA_PATH, WEIGHTS_PATH, SOLVER_PATH, LAYER_SIZES[LAYER], num_iterations=100, batch_size=25, save_every=20)
	# final_weights = g.get_weights() 
	# np.save('trained_weights', final_weights)

if __name__ == '__main__':
	main()