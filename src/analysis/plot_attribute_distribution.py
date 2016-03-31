import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def moveon(event):
    plt.close()

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def main():
	ATTRIBUTE = 18
	blob = np.load('./outs-fc6-100/blob-%d.dat.npy'%(ATTRIBUTE))

	print blob.shape

	# freqs = np.zeros()
	# selected_images = (np.abs(blob[:,0]) < 0.01)
	# selected_images_2 = np.logical_and((blob[:,0] < 0.01), (blob[:,0] > -0.01))

	# print np.sum(selected_images)
	for i in xrange(74):
		n, bins, patches = plt.hist(blob[:, i], 50, normed=1, facecolor='green', alpha=0.75)

		# print n.shape
		# print bins.shape
		# hist, bin_edges = np.histogram(n, density=True)
		# bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

		# p0 = [1., 0., 1.]

		# coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
		# hist_fit = gauss(bin_centres, *coeff)
		# plt.plot(bins[1:], n, label='Fitted data')
		# min_val = bins[np.argmax(n)]
		# max_val = bins[np.argmax(n)+1]

		# mode_images = np.logical_and((blob[:,0] <= max_val), (blob[:,0] => min_val))


		# fig = plt.figure()
		# cid = fig.canvas.mpl_connect('key_press_event', moveon)
		plt.show()
		# plt.savefig('./attribute_distributions/attribute_%d.png'%(i))

if __name__ == '__main__':
	main()