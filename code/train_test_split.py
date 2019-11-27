import numpy as np

def train_test_split(X, y, train_ratio = 0.7):
    
    if len(X) == len(y):
        train_size = round(len(y) * train_ratio)
        
        random_idx = np.arange(len(y))
        np.random.shuffle(random_idx)
        
        train_idx = random_idx[: train_size]
        test_idx = random_idx[train_size :]
        
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        return X_train, X_test, y_train, y_test
    
    else:
        print('X와 y의 길이가 맞지 않습니다.')