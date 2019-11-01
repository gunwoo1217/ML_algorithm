import numpy as np
import cvxopt as cvx

class svm_classifier:
    
    def fit(self, X_train, y_train):
        
        n_samples, n_feature = X_train.shape
        y_train = y_train.reshape(-1, 1)

        P = np.dot(X_train, X_train.T) * np.dot(y_train, y_train.T).astype('double')
        q = -np.ones(n_samples).astype('double')
        G = -np.eye(n_samples).astype('double')
        h = np.zeros(n_samples).astype('double')
        A = y_train.T.astype('double')
        
        P = cvx.matrix(P)
        q = cvx.matrix(q)
        G = cvx.matrix(G)
        h = cvx.matrix(h)
        A = cvx.matrix(A)
        b = cvx.matrix(0.0)
        
        sol = np.ravel(cvx.solvers.qp(P, q, G, h, A, b)['x'])
        
        lagrange_multiplier = sol[sol > 1e-6]
        sv_X_train = X_train[sol > 1e-6]
        sv_y_train = y_train.flatten()[sol > 1e-6]
        weight = np.dot(lagrange_multiplier * sv_y_train, sv_X_train)
        bias = np.dot(weight, sv_X_train[0]) - 1
        
        self.weight = weight
        self.bias = bias
    
    def predict(self, X_test):
        
        criteria = np.dot(X_test, self.weight) + self.bias > 0
        y_pred = criteria.astype(int)
        y_pred[y_pred == 0] = -1
        return y_pred
    
    def score(self, X_test, y_test):
        return np.mean(self.predict(X_test) == y_test)
