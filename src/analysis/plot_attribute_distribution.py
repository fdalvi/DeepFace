import numpy as np
import matplotlib.pyplot as plt

def moveon(event):
    plt.close()



def main():
	ATTRIBUTE = 1
	blob = np.load('./outs-fc6-100/blob-%d.dat.npy'%(ATTRIBUTE))

	print blob.shape

	# freqs = np.zeros()
	# selected_images = (np.abs(blob[:,0]) < 0.01)
	# selected_images_2 = np.logical_and((blob[:,0] < 0.01), (blob[:,0] > -0.01))

	# print np.sum(selected_images)
	for i in xrange(10):
		n, bins, patches = plt.hist(blob[:, i], 50, normed=1, facecolor='green', alpha=0.75)
		# min_val = bins[np.argmax(n)]
		# max_val = bins[np.argmax(n)+1]

		# mode_images = np.logical_and((blob[:,0] <= max_val), (blob[:,0] => min_val))


		# fig = plt.figure()
		# cid = fig.canvas.mpl_connect('key_press_event', moveon)
		plt.show()
		# plt.savefig('./attribute_distributions/attribute_%d.png'%(i))

if __name__ == '__main__':
	main()