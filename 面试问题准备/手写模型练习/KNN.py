import numpy as np
from collections import Counter
import math

class KNNClassifier:

    def __init__(self,k):
        assert k>=1,"k must a postive integeral"
        self.k = k
        self._X_train = None
        self._y_train = None
    
    def fit(self,X_train,y_train):
        self._X_train = X_train
        self._y_train = y_train

    
    def predict(self,X_predict):
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)


    def _predict(self,x):

        distance = sqrt(np.sum((self._X_train-x)**2,axis=0))
        nearest = np.argsort(distance)
        topK_y = self._y_train[nearest]

        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]
    
    def __repr__(self):
        return "KNN(k=%d)" % self.k

        
