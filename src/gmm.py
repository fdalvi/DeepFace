import numpy as np

class GMM():
	def __init__(self, means, vars_):
		self.means = means
		# self.covs = np.zeros((vars_.shape[0], vars_.shape[1], vars_.shape[1]))
		# for i in xrange(vars_.shape[0]):
		# 	print i
		# 	self.covs[i,:,:] = np.diag(vars_[i,:])
		self.covs = vars_
		self.num_components = means.shape[0]
		self.num_features = means.shape[1]
		self.weights = np.ones((self.num_components,))

	def sample(self, y):
		sampled_sum = np.zeros((self.num_features,))
		# out = np.zeros((self.num_components,))
		for i in xrange(self.num_components):
			if y[i] == 0:
				continue
			# out = np.random.multivariate_normal(self.means[i,:], self.covs[i,:,:])
			out = np.array([np.random.normal(self.means[i,j], self.covs[i,j]) for j in xrange(self.means.shape[1])])
			sampled_sum += self.weights[i] * out
		sampled_sum /= np.float(np.sum(y))

		return sampled_sum


def test():
	means = np.array([[2,3], [9, 10]])
	vars_ = np.array([[1,1], [2,2]])
	gmm = GMM(means, vars_)

	print gmm.sample(np.array([0,1]))

if __name__ == '__main__':
	test()