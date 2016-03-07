import numpy as np

class GMM():
	def __init__(self, means, covs):
		self.means = means
		self.covs = covs
		self.num_components = means.shape[0]
		self.num_features = means.shape[1]
		self.weights = np.ones((self.num_components,))

	def sample(self, y):
		sampled_sum = np.zeros((self.num_features,))
		for i in xrange(self.num_components):
			if y[i] == 0:
				continue
			out = np.random.multivariate_normal(self.means[i,:], self.covs[i, :, :])
			sampled_sum += self.weights[i] * out
		sampled_sum /= np.float(np.sum(y))

		return sampled_sum


def test():
	means = np.array([[2,3], [9, 10]])
	covs = np.array([[[1,0],[0,1]], [[2,0],[0,2]]])
	gmm = GMM(means, covs)

	print gmm.sample(np.array([0,1]))

if __name__ == '__main__':
	test()