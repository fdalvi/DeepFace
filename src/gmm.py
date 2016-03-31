import numpy as np
import util
import random
import os

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
		self.lr = 1e-8
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
		# target_feats: N x F
		# true_labels: N x 73 
		# means: 73 x F
		##forward pass 

		N, F = target_feats.shape
		sampled_vec_feats = np.dot(true_labels, (self.weights.reshape(-1,1)*self.means))

		# Non vectorized
		# sampled_feats = np.zeros((N,F))
		# for i in xrange(target_feats.shape[0]): 
		# 	# print true_labels[i,:].shape
		# 	# print true_labels[i,:].reshape(true_labels[i,:].shape[0], -1).shape
		# 	print '\tIteration %d/%d'%(i+1, target_feats.shape[0])
		# 	sampled_feats[i,:] = np.sum(true_labels[i,:].reshape(-1,1)*(self.weights.reshape(-1,1)*self.means), axis=0) #1 x F 
		# print np.sum(np.abs(sampled_feats - sampled_vec_feats))

		# Compute Loss
		loss = np.sum((target_feats - sampled_vec_feats)**2)
		loss /= float(N) 
		loss += 0.5 * self.reg * np.sum(self.weights**2)
		print "loss:", loss 

		# Backward pass + step 
		dweights = np.zeros((self.num_components,))
		for i in xrange(self.num_components): 
			# print '\t iteration %d/%d'%(i, self.num_components)
			dweights[i] = -2 * np.sum((target_feats - sampled_vec_feats) * true_labels[:,i].reshape(-1,1).dot(self.means[i,:].reshape(1,-1)))
			dweights[i] /= float(N)
			dweights[i] += self.reg * self.weights[i]
			self.weights[i] -= self.lr * dweights[i]

		return loss, dweights

	def gmm_loss(self, target_feats, true_labels, weights): 
		N, F = target_feats.shape
		sampled_vec_feats = np.dot(true_labels, (weights.reshape(-1,1)*self.means))

		# Compute Loss
		loss = np.sum((target_feats - sampled_vec_feats)**2)
		loss /= float(N) 
		loss += 0.5 * self.reg * np.sum(weights**2)

		return loss

	def get_weights(self): 
		return self.weights

	def load_weights(self, weights_path): 
		self.weights = np.load(weights_path)

	def train(self, layer, data_path, weights_path, solver_path, output_path, layer_dims, num_iterations=100, batch_size=25, save_every=20): 
		local_output_path = os.path.join(output_path, 'weights')
		cache = None
		images = util.get_image_names(data_path)
		check_gradient = False

		if not os.path.exists(local_output_path):
			os.makedirs(local_output_path)

		loss_history = []
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
			weights = self.weights.copy()
			loss, dweights = self.step(feats, labels)
			loss_history.append(loss)

			#check gradient
			# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
			if check_gradient: 
				f = lambda w: self.gmm_loss(feats, labels, w)
				util.grad_check_sparse(f, weights, dweights)

			if (i+1) % save_every == 0 or i == num_iterations-1: 
				print 'Saving weights...'
				np.save(os.path.join(local_output_path, 'weights_iter%d'%(i+1)), self.weights)
				np.save(os.path.join(local_output_path, 'loss_history'), np.array(loss_history))


def test():
	means = np.array([[2,3], [9, 10]])
	vars_ = np.array([[1,1], [2,2]])
	gmm = GMM(means, vars_)

	print gmm.sample(np.array([0,1]))

if __name__ == '__main__':
	test()