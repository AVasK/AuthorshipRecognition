### Pure Python Logistic Regression (simple) ###
import random
import numpy as np

"""
BRIEF LOOK AT STRUCTURES USED HERE:

W = [w0, w1, w2, ..., w_N] <N+1 dim.>
     ^   '-----^--------'
    bias    weights
            
    __          __
    |  |  |  |   |
X = |  x0 x1 ... |   <(M x N+1) dim.>
    |_ |  |  |  _|
       ^
    a single column
    is a single training vector

    
Y = [l1, ... , l_N]  <N dim.>
      ^
    label
    
"""

class LogisticRegression:
    
    def __init__(self, X_train = False, Y_train = False, activation = "sigmoid", alpha = 0.1, epochs = 1000):
        self.verbose = True
        self.X = np.array(X_train)
        if self.verbose: print(self.X)
        self.Y = np.array(Y_train)
        self.M, self.N = self.X.shape
        self.X = np.hstack((np.ones((self.M, 1)), self.X))
        self.M, self.N = self.X.shape
        if self.verbose: print(self.X)
        self.activation_type = activation
        self.alpha = alpha
        self.epochs = epochs
        
    def fit(self, X = False, Y = False, epochs = False):
        if epochs:
            self.epochs = epochs
            
        if X and Y:
            # dataset is given
            self.X = np.array(X)
            self.Y = np.array(Y)
        
            self.M, self.N = self.X.shape
            self.X = np.hstack((np.ones((self.M, 1)), self.X))
            
            if self.verbose: print(self.X, self.Y, self.M, self.N)
        
        # dataset given on initialization <OR>
        # on previous step
        
        # weights initialized to a [0, 1) float.
        self.W = np.array([random.random() for _ in range(self.N)]) 
        if self.verbose: print(self.W)
        self.train()

        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
            
    def loss(self, H):
        return (-self.Y * np.log(H) - (1 - self.Y) * np.log(1 - H)).mean()
            
    def update(self, epoch):
        Z = np.dot(self.X, self.W) # should be of shape (M,1)
        H = self.sigmoid(Z)
        gradient = np.dot(self.X.T, (H - self.Y)) / self.Y.size
        self.W -= self.alpha * gradient
        if self.verbose and epoch % 100 == 0:
            print(f"loss on epoch {epoch}:{self.loss(H)}")
            
    def train(self):
        for epoch in range(1, self.epochs):
            self.update(epoch)
        
    @staticmethod
    def add_ones(X_test):
        X_test = np.array(X_test)
        X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        return X_test    
            
    def predict_proba(self, X_test):
        X_test = LogisticRegression.add_ones(X_test)
        Z = np.dot(X_test, self.W.T) # should be of shape (M,1)
        H = self.sigmoid(Z)
        return H
        
    def predict(self, X_test, threshold = 0.5):
        return self.predict_proba(X_test) >= threshold
            
        
        
        
if __name__ == '__main__':
    X_train = [[1,2], [1,1], [2,1], [2,3], [3,3], [3,2]]
    model = LogisticRegression(X_train, [1,1,1,0,0,0])
    model.fit()
    print(model.predict([[1.5, 1.5]]))
    print(model.predict([[3, 3]]))
    