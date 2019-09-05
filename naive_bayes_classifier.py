import numpy as np

class naive_bayes_classifier:
        
    def fit(self, X_train, y_train):
        
        y_label_list, y_label_counts = np.unique(y_train, return_counts = True)
        var_label_parameter = []
        
        for var in range(X_train.shape[1]):
            label_parameter = []
            
            for label in range(len(y_label_list)):
                label_boolean = y_train == label
                mean = np.mean(X_train[label_boolean, var])
                std = np.std(X_train[label_boolean, var])
                label_parameter.append({'mean' : mean,
                                        'std' : std})
            
            var_label_parameter.append(label_parameter)
            
        self.var_label_parameter = var_label_parameter
        
    def predict(self, X_test):
        