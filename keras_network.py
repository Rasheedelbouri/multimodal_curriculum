#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:12:01 2019

@author: Rasheed el-Bouri
"""
import os    
#os.environ['THEANO_FLAGS'] = "device=cuda0"    
#import theano
#Stheano.config.floatX = 'float32'
import numpy as np
import matplotlib.pyplot as plt#
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from Allmyfunctions import readTable, Get_Mahalanobis, Get_H_features, Get_H_wardtypes, Standardise_data
import pickle
import argparse
import sys
class generateTrainingData():
    
    
    def __init__(self, source, red_feat, balanced_class, greyout):

        self.source = source
        self.balanced_class = balanced_class
        self.greyout = greyout
        
        if self.source == 'O':
            self.inputs = readTable('../../ORCHID_data/Clustering/testing_feats.csv', separator=',')
            self.labels = readTable('../../ORCHID_data/Clustering/testing_targets.csv', separator=',')
        elif self.source == 'H':
            self.inputs = Get_H_features()[1]
            self.inputs = self.inputs[self.inputs.initial_location.isin(list(range(75,183)))]
            self.labels = Get_H_wardtypes(pd.DataFrame(self.inputs.initial_location))

            self.inputs = self.inputs.drop('initial_location',axis=1)
            if red_feat == True:
                self.inputs = self.inputs[['Blood Culture', 'Cardiac Enzymes', 'Cross Match Blood', 'Pregnancy Test','diagnosis_code','frequent_flier']]

        elif self.source == 'M':
            train_x = pd.read_csv('mimic_data/mimic_train_in.csv', sep=',', index_col=0)
            val_x = pd.read_csv('mimic_data/mimic_val_in.csv', sep=',', index_col=0)
            test_x = pd.read_csv('mimic_data/mimic_test_in.csv', sep=',', index_col = 0)
            
            self.inputs = pd.concat([train_x, val_x], axis=0)
            self.inputs = pd.concat([self.inputs, test_x], axis=0)
            
            train_y = pd.read_csv('mimic_data/mimic_train_out.csv', sep=',', index_col=0)
            val_y = pd.read_csv('mimic_data/mimic_val_out.csv', sep=',', index_col=0)
            test_y = pd.read_csv('mimic_data/mimic_test_out.csv', sep=',', index_col=0)
#            
            self.labels = pd.concat([train_y, val_y], axis=0)
            self.labels = pd.concat([self.labels, test_y], axis=0)
            
        else:
            sys.exit("Must specify 'H' or 'O' or 'M'")


    def getDataSplit(self):
        uniques = pd.DataFrame(self.labels.iloc[:,0].unique()).sort_values(0).reset_index(drop=True)
        numbers = list(range(0,len(uniques)))
        
        for i in range(0,len(uniques)):
            self.labels.iloc[:,0][self.labels.iloc[:,0] == uniques[0][i]] = numbers[i]
            
        self.labels.columns = ['label']
        
        if self.source == 'O':
            self.inputs = self.inputs.drop(self.labels[self.labels['label'] == 8].index[0])
            self.labels = self.labels[self.labels['label'] != 8]
        
        elif self.source =='H':
             self.inputs = self.inputs.drop(self.labels[self.labels['label'] == 1].index)
             self.inputs = self.inputs.drop(self.labels[self.labels['label'] == 3].index)
             self.inputs = self.inputs.astype(float)
             self.inputs = Standardise_data(self.inputs)
             
             
             self.labels = self.labels[self.labels['label'] != 1]
             self.labels = self.labels[self.labels['label'] != 3]
             
             
             
             uniques = pd.DataFrame(self.labels.iloc[:,0].unique()).sort_values(0).reset_index(drop=True)
             uniques['count'] = uniques[0].map(self.labels.label.value_counts())


             if self.greyout == True:
                  for i in range(0,len(uniques)):
                      self.labels.iloc[:,0][self.labels.iloc[:,0] == uniques[0][i]] = numbers[i]
                  uniques = pd.DataFrame(self.labels.iloc[:,0].unique()).sort_values(0).reset_index(drop=True)
                  uniques['count'] = uniques[0].map(self.labels.label.value_counts())                 
            
            
             if self.balanced_class == True:
                 
                 label = dict()
                 ins = dict()
                 
                 for i in range(0,len(uniques[0])):
                     label[i] = self.labels[self.labels.label == uniques[0][i]]
                     label[i] = label[i].sample(frac=(min(uniques['count'])/len(label[i])))
                     
                     ins[i] = self.inputs.loc[label[i].index]
                     
                 self.inputs = pd.DataFrame()
                 self.labels = pd.DataFrame()
                 
                 for i in range(0,len(label)):
                     self.inputs = pd.concat([self.inputs, ins[i]], axis=0)
                     self.labels = pd.concat([self.labels, label[i]], axis=0)

                    

             
        elif self.source == 'M':
            self.inputs = self.inputs.astype(float)
             

              
            
        
        uniques = pd.DataFrame(self.labels.iloc[:,0].unique()).sort_values(0).reset_index(drop=True)

        
        y = preprocessing.label_binarize(self.labels, classes=list(range(0,len(uniques))))
        
        
        train_x, test_x, train_y, test_y = train_test_split(self.inputs, y, train_size=0.6, stratify = self.labels) 
                                                            
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=0.7, random_state=1)
        
        return(train_x, train_y, val_x, val_y, test_x, test_y, uniques)
        
    def encodedInputs(self, train_x, val_x):
        seed = 7
        np.random.seed(seed) # setting initial seed for reproducable results
        
        if train_x.shape[1] < 10:
            encoder = Sequential() # begin a sequential model
            encoder.add(Dense(train_x.shape[1], input_dim=train_x.shape[1], activation='tanh'))
            encoder.add(Dense(train_x.shape[1], activation = 'tanh', W_regularizer = regularizers.l2(1e-4)))
            encoder.add(Dense(train_x.shape[1], activation = 'tanh', W_regularizer = regularizers.l2(1e-4)))
            encoder.add(Dropout(0.1))
            encoder.add(Dense(train_x.shape[1], activation = None))
        
        
        elif train_x.shape[1] > 50:        # create model
            encoder = Sequential() # begin a sequential model
            encoder.add(Dense(train_x.shape[1], input_dim=train_x.shape[1], activation='tanh'))
            encoder.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
            encoder.add(Dense(train_x.shape[1], activation = 'tanh', W_regularizer = regularizers.l2(1e-4)))
            encoder.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
            encoder.add(Dropout(0.1))
            encoder.add(Dense(train_x.shape[1], activation = None))
            
        else:
            encoder = Sequential()
            encoder.add(Dense(train_x.shape[1], input_dim=train_x.shape[1], activation='linear'))

        sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
                #model.add_loss(custom_objective)
        encoder.compile(loss='mean_squared_error', 
                          optimizer= sgd, 
                          metrics=['accuracy'])# full_number_accuracy])
        
        if type(train_x) == dict:
            history = encoder.fit(train_x[sorted(train_x.keys())[-1]], train_x[sorted(train_x.keys())[-1]], 
                          epochs=100, 
                          batch_size=100,
                          shuffle = True,
                          validation_data = (val_x,val_x),
                          verbose = 1).history
        else:
            history = encoder.fit(train_x, train_x, 
                          epochs=100, 
                          batch_size=100,
                          shuffle = True,
                          validation_data = (val_x,val_x),
                          verbose = 1).history
        
        return(encoder, history)