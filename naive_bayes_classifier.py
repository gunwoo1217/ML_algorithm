import numpy as np

class naive_bayes_classifier:
    print('가우시안 정규분포를 가정한 나이브베이즈 모형입니다. 이항분포 다항분포에 대한 나이브베이즈 모형은 나중에...')
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        y_label_list, y_label_counts = np.unique(y_train, return_counts = True)
        self.y_label_list = y_label_list
        self.y_label_counts = y_label_counts
        
        parameter = []
        
        for label in y_label_list:
            label_boolean = (self.y_train == label).flatten()
            mean_var_list = []
            
            for v in range(self.X_train.shape[1]):
                mean = np.mean(X_train[label_boolean, v])
                var = np.var(X_train[label_boolean, v])
                mean_var_list.append({'mean' : mean,
                                      'var' : var})
        
            parameter.append(mean_var_list)
            
        self.parameter = parameter
        return self.parameter
        
    def predict(self, X_test):
        
        self.X_test = X_test
        
        y_pred = []
        
        for x in range(len(self.X_test)):
            posterior_list = []
            
            for label_loc in range(len(self.y_label_list)):
                prior = self.y_label_counts[label_loc] / len(self.y_train)
                likelihood = 1
                
                for v in range(self.X_train.shape[1]):
                    mean = self.parameter[label_loc][v]['mean']
                    var = self.parameter[label_loc][v]['var']
                    
                    likelihood *= np.exp(-((self.X_test[x, v] - mean) ** 2) / (2 * var)) / (np.sqrt(2 * var))
                
                posterior_list.append(prior * likelihood)
            
            max_loc = np.argmax(posterior_list)
            y_pred.append(self.y_label_list[max_loc])
        
        return np.array(y_pred)
    
    def score(self, X_test, y_test):
        
        return np.mean(self.predict(X_test) == y_test)
