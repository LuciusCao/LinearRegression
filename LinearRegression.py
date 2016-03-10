import numpy as np

class LinearRegression:
	def create_data(self,number_of_examples, number_of_features):
		X = np.random.randint(0,100,(number_of_examples,number_of_features))
		y = np.random.randint(-100,1000,(number_of_examples,1))
		theta = np.zeros((number_of_features+1,1))
		X = np.matrix(X)
		y = np.matrix(y)
		theta = np.matrix(theta)
		return X, y, theta

	def feature_normalize(self,X):
		mean = X.mean(0)
		std = X.std(0)
		normalized_features = (X-mean)/std
		return normalized_features

	def add_bias(self,X):
		bias_feature = np.ones((len(y),1))
		new_X = np.column_stack((bias_feature,X))
		return new_X

	def cost_function(self,X, y, theta, reg_rate = 0):
		m = len(y)
		cost = float((1/(2*m))*sum((X.dot(theta)-y).A**2)) + float(reg_rate * sum((theta[1:,]).A ** 2))
		return cost

	def gradient_descent(self,X, y, theta, reg_rate = 0):
		m = len(y)
		grad = np.matrix((1/m)*sum((X.dot(theta)-y).A * X.A))
		grad = grad.T
		return grad

if __name__ == '__main__':
	lr = LinearRegression()
	X, y, theta = lr.create_data(10000,50)
	X = lr.feature_normalize(X)
	X = lr.add_bias(X)
	cost = []
	for i in range(500):
		grad = lr.gradient_descent(X,y,theta,1)
		theta = theta - 0.003 * grad
		cost.append(lr.cost_function(X,y,theta))
	print(cost[-1])





