import numpy as np
import util
import random

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
		self.lr = 5e-4
		self.reg = 1e-5

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

	def step(self, target_feats, true_labels):
		#target_feats: N x F
		#true_labels: N x 73 
		#means: 73 x F
		##forward pass 

		# print 'target_feats: ', target_feats.shape
		# print 'true_labels: ', true_labels.shape
		N, F = target_feats.shape
		# print "num_attr, F:", self.means.shape[0], F
		sampled_feats = np.zeros((N,F))
		sampled_vec_feats = np.dot(true_labels, (self.weights.reshape(-1,1)*self.means))
		# for i in xrange(target_feats.shape[0]): 
		# 	# print true_labels[i,:].shape
		# 	# print true_labels[i,:].reshape(true_labels[i,:].shape[0], -1).shape
		# 	print '\tIteration %d/%d'%(i+1, target_feats.shape[0])
		# 	sampled_feats[i,:] = np.sum(true_labels[i,:].reshape(-1,1)*(self.weights.reshape(-1,1)*self.means), axis=0) #1 x F 
		# print np.sum(np.abs(sampled_feats - sampled_vec_feats))
		loss = np.sum((target_feats - sampled_feats)**2)
		loss /= float(N) 
		loss += 0.5 * self.reg * np.sum(self.weights**2)
		print "loss:", loss 

		##backward pass + step (need to add regularization)
		dweights = np.zeros((self.num_components,))
		# vec_dweights = -2 * np.dot(  (target_feats - sampled_vec_feats), np.dot(true_labels, np.tile(self.means, (N,1)))
		# vec_dweights = np.sum(true_labels * (-2 * np.dot((target_feats - sampled_vec_feats), self.means.T)), axis=0)
		# updated_weights = self.weights - self.lr * vec_dweights 
		for i in xrange(self.num_components): 
			# print '\t iteration %d/%d'%(i, self.num_components)
			dweights[i] = -2 * np.sum((target_feats - sampled_feats) * true_labels[:,i].reshape(-1,1).dot(self.means[i,:].reshape(1,-1)))
			dweights[i] /= float(N)
			dweights[i] += self.reg * self.weights[i]
			self.weights[i] -= self.lr * dweights[i]
		# print np.sum(np.abs(dweights - vec_dweights))
		# print np.sum(np.abs(updated_weights - self.weights))

		return loss

	def train(self, layer, data_path, weights_path, solver_path, layer_dims, num_iterations=100, batch_size=25): 
		cache = None
		images = util.get_image_names(data_path)

		for i in xrange(num_iterations): 
			print 'Train iteration %d/%d'%(i+1, num_iterations)
			# Sample images
			print '\tSampling %d images...'%(batch_size)
			batch = random.sample(images, batch_size)

			# Run forward pass
			print '\tExtracting activations...'
			labels, feats, cache = util.extract_batch_activations(layer, data_path, weights_path, solver_path, layer_dims, batch, cache)
			feats = feats.reshape((feats.shape[0], -1))

			# Perform BGD step
			print '\tStepping...'
			print '\t',
			loss = self.step(feats, labels)


def test():
	means = np.array([[2,3], [9, 10]])
	vars_ = np.array([[1,1], [2,2]])
	gmm = GMM(means, vars_)

	print gmm.sample(np.array([0,1]))

if __name__ == '__main__':
	test()