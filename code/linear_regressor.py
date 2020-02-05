import numpy as np

class linear_regressor:
        
    def __init__(self, bias = True):
        '''
        편향의 유무를 설정할 수 있다.
        '''
        self.bias = bias
    
    def fit(self, X_train, y_train):
        '''
        회귀계수를 구할 수 있다.
        '''
        if X_train.ndim == 1:
            X_train = X_train.reshape(len(X_tarin), 1)
        
        if self.bias == True:
            one_vector = np.ones([len(X_train), 1])
            self.X_train = np.hstack([one_vector, X_train])
        else:
            self.X_train = X_train        
        self.y_train = y_train.reshape(len(y_train), 1)
        
        mat1 = np.linalg.inv(np.dot(self.X_train.T, self.X_train))
        mat2 = np.dot(self.X_train.T, self.y_train)
        weight = np.dot(mat1, mat2)
        self.weight = weight
        
        return weight
        
    def predict(self, X_test):
        '''
        예측값을 반환한다.
        '''
        
        if self.bias == True:
            one_vector = np.ones([len(X_test), 1])
            self.X_test = np.hstack([one_vector, X_test])
        else:
            self.X_test = X_test
        
        y_pred = np.dot(self.X_test, self.weight)
        
        return y_pred
