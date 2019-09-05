import numpy as np

class logistic_regressor:
    
    def __init__(self, bias = True, threshold = 0.5):
        self.bias = bias
        self.threshold = threshold
    
    def fit(self, X_train, y_train):
        
        if self.bias == True:
            one_vector = np.ones([len(X_train), 1])
            self.X_train = np.hstack([one_vector, X_train])            
        else:
            self.X_train = X_train        
        self.y_train = y_train
        
        mat1 = np.linalg.inv(np.dot(self.X_train.T, self.X_train))
        mat2 = np.dot(self.X_train.T, self.y_train)
        
        weight = np.dot(mat1, mat2)
        self.weight = weight
        
    def predict(self, X_test):
        
        if self.bias == True:
            one_vector = np.ones([len(X_test), 1])
            self.X_test = np.hstack([one_vector, X_test])
        else:
            self.X_test = X_test
        
        y_linear_pred = np.dot(self.X_test, self.weight)
        sigmoid = lambda x : 1 / (1 + np.exp(-x))        
        y_sigmoid_pred = sigmoid(y_linear_pred)
        y_boolean_pred = y_sigmoid_pred > self.threshold
        y_pred = y_boolean_pred.astype(np.int)
        self.y_pred = y_pred
        
        return y_pred
    
    def score(self, X_test, y_test):
        
        return np.mean(self.predict(X_test) == y_test)