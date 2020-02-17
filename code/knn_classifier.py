import numpy as np

class knn_classifier:
    
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
            
            label = self.y_train[near_idx][:self.k]
            label_idx, label_count = np.unique(label, return_counts = True)
            most_counts_label = label_idx[np.argmax(label_count)]
            y_pred.append(most_counts_label)
        
        return np.array(y_pred)
    
    def score(self, X_test, y_test):
        
        return np.mean(self.predict(X_test) == y_test)
