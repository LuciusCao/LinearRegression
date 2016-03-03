import numpy as np

class LinearRegression:
	def __init__(self):
		self.learning_rate = 0.01
		self.reg_rate = 0.003
		self.max_iter = 500

	def data_loader():
		'''including init X and y and theta'''
		pass

	def feature_normalize(self, X):
		mean = X.mean(0)
		std = X.std(0)
		normalized_features = (X-mean)/std
		return normalized_features

	def cost_function(self, X, y, theta):
		m = len(y)
		cost = (1/(2*m))*sum((X.dot(theta)-y).A**2)
		return cost

	def gradient_descent(self, X, y, theta):
		m = len(y)
		grad = 