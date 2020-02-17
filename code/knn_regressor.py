import numpy as np

class knn_regressor:
    
    def __init__(self, k = 3):
        self.k = k
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        self.X_test = X_test
        
        y_pred = []
        
        for test_point in self.X_test:
            diff_square_list = []
            
            for train_point in self.X_train:
                diff_vector = test_point - train_point
                diff_square = np.dot(diff_vector, diff_vector)
                diff_square_list.append(diff_square)
            
            diff_square_list = np.array(diff_square_list)
            near_idx = np.argsort(diff_square_list)
            
            y_pred.append(np.mean(self.y_train[near_idx][:self.k]))
        
        return np.array(y_pred)
