#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:39:06 2021

@author: kebl4170
"""

from model import buildNetwork

class trainingRoutine():
    
    def __init__(self, curricBatches, curricOuts, depthFirst):
        
        assert isinstance(curricBatches, dict)
        assert isinstance(curricOuts, dict)
        assert isinstance(depthFirst, bool)
        
        self.curricBatches = curricBatches
        self.curricOuts = curricOuts
        self.depthFirst = depthFirst
    
    def buildModel(self, train_x, uniques):
        bn = buildNetwork(source='H', seed = 0, hidden_layers=2, hidden_nodes=50, temp=1,
                          dropout=0.2,activation='relu', batchnorm=True, numepochs=100, 
                          batchsize=30,curriculum_batches = 10, curriculum_recursion = 1,
                          q_net = False,act_net = False, crit_net=False)
        
        network = bn.build(train_x, uniques, conv=False)
        network = bn.compiler(network, q_net=False, actor=False)
        
        return network
    
    def trainNetwork(self, train_x, val_x, val_y, uniques):
        
        valAccs = []
        
        network = self.buildModel(train_x, uniques)
        if self.depthFirst:
            for i in range(len(self.curricBatches)):
                for repetition in range(500):
                    for j in range(len(self.curricBatches[i])):
                        network.train_on_batch(self.curricBatches[i][j], self.curricOuts[i][j])
                        valAccs.append(network.test_on_batch(val_x, val_y)[1])
        else:
            for i in range(len(self.curricBatches[0])):
                for repetition in range(500):
                    for j in range(len(self.curricBatches)):
                        network.train_on_batch(self.curricBatches[j][i], self.curricOuts[j][i])
                        valAccs.append(network.test_on_batch(val_x, val_y)[1])

        return valAccs
        