# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:03:23 2020

@author: rashe
"""

import numpy as np
import pandas as pd
from data_loading import loadData

class clusterGenerator():
    
    def __init__(self, data, numbatches, num_clusters, multimodal=True, mix_type='gaussian'):
        
        assert(type(multimodal) == bool)
        assert(type(numbatches) == int)
        assert(type(num_clusters) == int)
        
                
        self.numbatches = numbatches
        self.num_clusters = num_clusters
        self.multimodal = multimodal
        self.mixtype = mix_type
        
      
        self.train_x = self.toDataFrame(data[0])
        self.train_y = self.toDataFrame(data[1])
        self.val_x = self.toDataFrame(data[2])
        self.val_y = self.toDataFrame(data[3])
        self.test_x = self.toDataFrame(data[4])
        self.test_y = self.toDataFrame(data[5])
        self.uniques = self.toDataFrame(data[6])

        if self.mixtype == "class":
            self.num_clusters = len(self.uniques)

    @staticmethod
    def toDataFrame(array):
        return pd.DataFrame(array)


    def createClusters(self):
        
        assert(self.mixtype.lower() in ('gaussian', 'k-means', 'class'))
        
        if self.mixtype.lower() == 'gaussian':
            from sklearn.mixture import GaussianMixture
            gm = GaussianMixture(n_components = self.num_clusters, covariance_type='full', n_init=100)
            predictions = gm.fit_predict(self.train_x)
            return predictions
        if self.mixtype.lower() == 'k-means':
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters = self.num_clusters, n_init=100)
            predictions = km.fit_predict(self.train_x)
            return predictions
        if self.mixtype.lower() == 'class':
            return np.argmax(np.array(self.train_y), axis=1)
        
        
    def separateClusters(self):
        cl = pd.DataFrame(self.createClusters())
        batches = dict()
        outs = dict()
        for i in range(self.num_clusters):
            batches[i] = self.train_x.iloc[cl[cl[0] == i].index]
            outs[i] = self.train_y.iloc[batches[i].index]
            assert(len(batches[i]) == len(outs[i]))
            batches[i] = batches[i].reset_index(drop=True)
            outs[i] = outs[i].reset_index(drop=True)
        
        return batches, outs
        

if __name__ == "__main__":
    data = loadData('iris') 
    cg = clusterGenerator(data,10, 3, multimodal=True, mix_type='k-means') 
    b,o = cg.separateClusters()      