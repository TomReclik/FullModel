import numpy as np
import scipy
from sklearn import cluster, datasets, mixture
import os

class Localizer:
    def __init__(self,eps, min_samples, theta):
        self.eps = eps
        self.min_samples = min_samples
        self.theta = theta

    def predict(self,SEM):
        """
        Predict the positions of damage sites
        Input: Array representing grayscale image
        Output: Centroids of proposed damage sites
        """

        size_x,size_y = SEM.shape
        pos = []
        for x in range(size_x):
            for y in range(size_y):
                if(SEM[x,y]<=self.theta):
                    pos.append((x,y))
        positions = np.array(pos)

        dbscan_dataset1 = cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean').fit_predict(positions)
        centroids = np.zeros((len(set(dbscan_dataset1))-1,2))
        
        for i in range(len(dbscan_dataset1)):
            if dbscan_dataset1[i]!=-1:
                centroids[dbscan_dataset1[i]] = centroids[dbscan_dataset1[i]] + positions[i]
        for i in range(len(centroids)):
                centroids[i] = centroids[i] / sum(dbscan_dataset1==i)

        return centroids
