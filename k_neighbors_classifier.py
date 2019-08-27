#%%
import numpy as np
import pandas as pd

#%%
class k_neighbors_classifier:
    import numpy as np
    import pandas as pd
    
    def __init__(self, k = 3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        self.X_test = X_test
        pred_list = []
        
        for test_point in self.X_test:
            dist_square_list = []
            
            for train_point in self.X_train:
                diff_vector = test_point - train_point
                dist_square = np.dot(diff_vector, diff_vector)
                dist_square_list.append(dist_square)
            
            dist_square_df = pd.DataFrame({'y_label' : self.y_train,
                                           'dist_square' : dist_square_list})
            dist_square_df = dist_square_df.sort_values('dist_square')        
            label_index, label_counts = np.unique(dist_square_df['y_label'][: self.k + 1], return_counts = True)
            most_counts_label = label_index[np.argmax(label_counts)]
            pred_list.append(most_counts_label)
        
        return pred_list
    
    def score(self, X_test, y_test):
        return np.mean(knn.predict(X_test) == y_test)