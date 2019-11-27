import numpy as np
import cvxopt as co

class svm_classifier:
    '''
    이진분류만 가능하며 라벨값은 0, 1로 받는다.
    슬랙변수를 추가한 모형(soft SVM)은 추후에 만들 예정
    '''
    
    def fit(self, X_train, y_train):
        
        n_samples, n_feature = X_train.shape
        y_train[y_train == 0] = -1
        y_train = y_train.reshape(-1, 1)

        P = np.dot((X_train.T * y_train.T).T, (X_train.T * y_train.T)).astype('double')
        q = -np.ones(n_samples).astype('double')
        G = -np.eye(n_samples).astype('double')
        h = np.zeros(n_samples).astype('double')
        A = y_train.T.astype('double')
        
        P = co.matrix(P)
        q = co.matrix(q)
        G = co.matrix(G)
        h = co.matrix(h)
        A = co.matrix(A)
        b = co.matrix(0.0)

        sol = np.ravel(co.solvers.qp(P, q, G, h, A, b)['x'])
        lagrange_multiplier = sol[sol > 1e-5]
        sv_X_train = X_train[sol > 1e-5]
        sv_y_train = y_train.flatten()[sol > 1e-5]
        weight = (lagrange_multiplier * sv_y_train * sv_X_train.T).sum(axis = 1)
        
        if sv_y_train[0] == -1:    
            bias = np.dot(weight, sv_X_train[0]) + 1
        elif sv_y_train[0] == 1:
            bias = np.dot(weight, sv_X_train[0]) - 1
        else:
            pass
        
        y_train[y_train == -1] = 0
        
        self.weight = weight
        self.bias = bias
    
    def predict(self, X_test):
        
        criteria = np.dot(X_test, self.weight) - self.bias > 0
        y_pred = criteria.astype(int)
        return y_pred
    
    def score(self, X_test, y_test):
        return np.mean(self.predict(X_test) == y_test)