import numpy as np
import os
import random
import caffe
import matplotlib.pyplot as plt
import util

DATA_PATH = '../data/eval_set/images_cropped/'
WEIGHTS_PATH = './snapshots/_iter_42000.caffemodel'
SOLVER_PATH = './DeepFaceNetDeploy.prototxt'
LAYER = 'fc6'
OUTPUT_PATH = './outs-%s'%(LAYER)

def main(): 
	# def extract_activations(layer, data_path, weights_path, solver_path, output_path, batch_size=25):
	util.extract_activations(LAYER, DATA_PATH, WEIGHTS_PATH, SOLVER_PATH, OUTPUT_PATH)


if __name__ == '__main__':
	main()
