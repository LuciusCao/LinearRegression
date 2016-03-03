import numpy as np

class LinearRegression:
	def __init__():
		learning_rate = 0.01
		reg_rate = 0.003
		max_iter = 500

	def data_loader():
		'''including init X and y and theta'''
		pass

	def feature_normalize(X):
		mean = X.mean(0)
		std = X.std(0)
		normalized_features = (X-mean)/std
		return normalized_features

	def add_bias(X):
		bias_feature = np.ones((len(y),1))
		new_X = np.column_stack((bias_feature,X))
		return new_X

	def cost_function(X, y, theta):
		m = len(y)
		cost = (1/(2*m))*sum((X.dot(theta)-y).A**2)
		return cost

	def gradient_descent(X, y, theta):
		m = len(y)
		grad = (1/m)*sum((X.dot(theta)-y).A * X.A)
		grad = grad.T
		return grad
