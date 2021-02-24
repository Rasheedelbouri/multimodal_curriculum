#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:12:01 2019

@author: kebl4170
"""
import os    
#os.environ['THEANO_FLAGS'] = "device=cuda0"    
#import theano
#Stheano.config.floatX = 'float32'
from keras.models import Sequential
from keras.layers import Dense, Lambda, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import keras.backend as K
import numpy as np
from keras.layers import Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt#
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from Allmyfunctions import readTable, Get_Mahalanobis, Get_H_features, Get_H_wardtypes, Standardise_data
import pickle
import argparse
from keras.models import load_model
from keras.engine.topology import Layer
import sys
import tensorflow as tf
from keras.losses import categorical_crossentropy
#from mine import MINE

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
                #self.inputs = self.inputs[['Blood Culture', 'Cardiac Enzymes', 'Cross Match Blood', 'diagnosis_code']]
                #self.inputs = self.inputs.drop(['subject_id', 'row_id', 'hadm_id', 'edattendance_id'], axis=1)

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
# # 
# =============================================================================
#         for i in range(0,len(uniques)):
#              self.labels.ix[:,0][self.labels.ix[:,0] == uniques[0][i]] = numbers[i]
# =============================================================================
# # =============================================================================
# =============================================================================
        
        y = preprocessing.label_binarize(self.labels, classes=list(range(0,len(uniques))))
        
        
        train_x, test_x, train_y, test_y = train_test_split(self.inputs, y, train_size=0.6, stratify = self.labels) 
                                                            
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=0.7, random_state=1)
        
        return(train_x, train_y, val_x, val_y, test_x, test_y, uniques)
        
    def encodedInputs(self, train_x, val_x):
        seed = 7
        np.random.seed(seed) # setting initial seed for reproducable results
        #temp= 1 # setting temperature of softmax functipn
        
        if train_x.shape[1] < 10:
            encoder = Sequential() # begin a sequential model
            encoder.add(Dense(train_x.shape[1], input_dim=train_x.shape[1], activation='tanh'))
            #encoder.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
            encoder.add(Dense(train_x.shape[1], activation = 'tanh', W_regularizer = regularizers.l2(1e-4)))
            encoder.add(Dense(train_x.shape[1], activation = 'tanh', W_regularizer = regularizers.l2(1e-4)))
            #encoder.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
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
# =============================================================================
#             encoder.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
#             encoder.add(Dense(train_x.shape[1], activation = 'tanh', W_regularizer = regularizers.l2(1e-4)))
#             encoder.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))
#             encoder.add(Dropout(0.3))
# =============================================================================
            #encoder.add(Dense(train_x.shape[1], activation = None))
        
        #model.add(Lambda(lambda x: x / temp))
        #model.add(Activation(self.val_y.shape[1], activation='softmax'))
            #student_model.add(Lambda(lambda x: x / temp))
        #model.add(Activation('softmax'))
        
            # Compile model
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
        
        
class Hadamard(Layer):

    def __init__(self, **kwargs):
        super(Hadamard, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,) + (train_x.shape[1],),
                                      initializer='uniform',
                                      trainable=True)
        super(Hadamard, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        print(x.shape, self.kernel.shape)
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return input_shape        

    


class buildNetwork():
    
    def __init__(self, seed = 7, hidden_layers=2, hidden_nodes=100, temp=1, dropout=0.2,\
                 activation='relu', batchnorm=True, sto_mini_batch_comp = True, numepochs=100, batchsize=30 \
                 ,curriculum_batches = 10, curriculum_recursion = 1):
        self.seed = seed
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.temp = temp
        self.dropout = dropout
        self.activation = activation
        self.batchnorm = batchnorm
        self.numepochs = numepochs
        self.batchsize = batchsize
        self.curriculum_batches = curriculum_batches
        self.curriculum_recursion = curriculum_recursion
        self.sto_mini_batch_comp = sto_mini_batch_comp
        

    def build(self, train_x, uniques):
        np.random.seed(self.seed) # setting initial seed for reproducable results
        temp = self.temp # setting temperature of softmax functipn
        
        model = Sequential() # begin a sequential model
        #model.add(Hadamard(input_shape=([82])))
        #model.add(Activation('softmax'))
        model.add(Dense(self.hidden_nodes, input_dim=train_x.shape[1], activation=self.activation))
        if self.batchnorm == True:
            model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

        for i in range(0, self.hidden_layers):
            model.add(Dense(self.hidden_nodes, activation= self.activation, W_regularizer = regularizers.l2(1e-3)))
            model.add(Dropout(self.dropout))
            if self.batchnorm == True:
                model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
        
        
        if len(uniques)<=2:
            model.add(Dense(len(uniques)-1, activation='sigmoid'))
        else:
            model.add(Dense(len(uniques), activation='softmax'))
            model.add(Lambda(lambda x: x / temp))
            #model.add(Dense(1, activation=None))
        
        model.summary()
        
        return(model)
        
    def my_loss(self, y_true, y_pred):
            weights = pd.DataFrame(self.model.layers[0].get_weights()[0])
            weights = np.array(weights)
            softmaxed = np.exp(weights/0.5)/(sum(np.exp(weights/0.5).T))
            
            entropy = -sum((softmaxed*np.log(softmaxed)).T)
            
            cat_cross = categorical_crossentropy(y_true, y_pred)
            
            output = cat_cross + tf.convert_to_tensor(entropy)
            #output = K.categorical_crossentropy(y_true, y_pred) + entropy
            return(output)
        
        
    def compiler(self, model,source, q_net):
        sgd = SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False)
        if q_net == True:
            model.compile(loss='mean_squared_error', #self.my_loss, 
                  optimizer= 'adam', 
                  metrics=['accuracy'])
        else:
            if source=='M':
                model.compile(loss='binary_crossentropy', 
                              optimizer= sgd, 
                              metrics=['accuracy'])#self.my_loss,
            else:
                model.compile(loss='categorical_crossentropy', #self.my_loss, 
                          optimizer= sgd, 
                          metrics=['accuracy'])
        
        return(model)
        
        
    def train(self, model, train_x, train_y, val_x, val_y):
        
        if self.sto_mini_batch_comp == True:
            train_x, train_y = pd.DataFrame(train_x).reset_index(drop=True), pd.DataFrame(train_y).reset_index(drop=True)
            training_performance=pd.DataFrame()
            validation_performance = pd.DataFrame()
            for i in range(0, self.numepochs):
                batchin = train_x.sample(frac=1/(train_x.shape[0]/50))
                indices = batchin.index
                batchout = train_y.loc[batchin.loc[indices].index]
                training_performance = pd.concat([training_performance, pd.DataFrame(model.train_on_batch(batchin, batchout))], axis=1)
                validation_performance = pd.concat([validation_performance, pd.DataFrame(model.test_on_batch(val_x, val_y))],axis=1)
                print(i)
            
            return(model, training_performance, validation_performance)
    
        else:
            history = model.fit(train_x, train_y, 
                  epochs=self.numepochs, 
                  batch_size=self.batchsize,
                  shuffle = True,
                  validation_data = (val_x,val_y),
                  verbose = 1).history
            return(model, history)

        
    def train_curriculum(self, model, train_x, train_y, val_x, val_y, numberbatches, curric_recursion):
        
        numbatches = numberbatches
        
        Mahas, features = Get_Mahalanobis(train_x)
        del features        
        Mahas = pd.DataFrame(Mahas).sort_values(0,ascending=True) ####### CHANGE MADE
        
        train_x, train_y = pd.DataFrame(train_x).reset_index(drop=True), pd.DataFrame(train_y).reset_index(drop=True)
        
        k = int(len(Mahas)/numbatches)

        batches = dict()
        outs = dict()
        for i in range(0, numbatches):
            indices = Mahas[0 : int(k*(i+1))].index
            batches[i] = train_x.loc[Mahas.loc[indices].index]
            outs[i] = train_y.loc[Mahas.loc[indices].index]
            
        
        training_performance = pd.DataFrame()
        validation_performance = pd.DataFrame()
        for j in range(0,len(batches)): ##########CHANGE MADE
            for i in range(0,curric_recursion):
                training_performance = pd.concat([training_performance, pd.DataFrame(model.train_on_batch(batches[j], outs[j]))], axis=1)
                validation_performance = pd.concat([validation_performance, pd.DataFrame(model.test_on_batch(val_x, val_y))],axis=1)
# =============================================================================
#                 if validation_performance.iloc[-1].iloc[-1] == max(validation_performance.iloc[-1]):
#                     model.save('Curriculum_winning_model.h5')
# =============================================================================
                
        return(model, training_performance, validation_performance, batches, outs)

        
        
    def extractEmbedding(self, curriculum, bandit, model, inputs, labels):
        
        if curriculum == False:
                
            get_hidden_layer_output = K.function([model.layers[0].input],
                                       [model.layers[len(model.layers)-1].output])
            layer_output = pd.DataFrame(get_hidden_layer_output([inputs])[0])
     
            layer_output.to_csv('finalhiddenlayer.csv',',', index=False, header=False)
    
            labels = pd.DataFrame(np.argmax(labels, axis=1))
    
            labels.to_csv('hiddenlayerlabels.csv', index=False, header=False)
        
        elif curriculum ==True and bandit == False:
        
            get_hidden_layer_output = K.function([model.layers[0].input],
                                       [model.layers[len(model.layers)-1].output])
            layer_output = pd.DataFrame(get_hidden_layer_output([inputs])[0])
     
            layer_output.to_csv('curric_finalhiddenlayer.csv',',', index=False, header=False)
    
            labels = pd.DataFrame(np.argmax(labels, axis=1))
    
            labels.to_csv('curric_hiddenlayerlabels.csv', index=False, header=False)   
        else:
            get_hidden_layer_output = K.function([model.layers[0].input],
                                       [model.layers[len(model.layers)-1].output])
            layer_output = pd.DataFrame(get_hidden_layer_output([inputs])[0])
     
            layer_output.to_csv('bandit_finalhiddenlayer.csv',',', index=False, header=False)
    
            labels = pd.DataFrame(np.argmax(labels, axis=1))
    
            labels.to_csv('bandit_hiddenlayerlabels.csv', index=False, header=False)
            
            
        
    def plotResults(self, hist, comparison):
        
            accs = pd.DataFrame()
            val_accs = pd.DataFrame()
            losses = pd.DataFrame()
            val_losses = pd.DataFrame()
            
            for i in range(0,5):
                accs = pd.concat([accs, pd.DataFrame(hist[i]['acc']).T],axis=0)
                val_accs = pd.concat([val_accs, pd.DataFrame(hist[i]['val_acc']).T],axis=0)
                losses = pd.concat([losses, pd.DataFrame(hist[i]['loss']).T],axis=0)
                val_losses = pd.concat([val_losses, pd.DataFrame(hist[i]['val_loss']).T],axis=0)
                
            uppers = np.zeros((4, self.numepochs))
            lowers = np.zeros((4, self.numepochs))
            accs , val_accs, losses, val_losses = pd.DataFrame(accs), pd.DataFrame(val_accs), pd.DataFrame(losses), pd.DataFrame(val_losses)
            
            
            for i in range(0, self.numepochs):
                uppers[0][i] = max(accs[i])
                uppers[1][i] = max(val_accs[i])
                uppers[2][i] = max(losses[i])
                uppers[3][i] = max(val_losses[i])
                
                lowers[0][i] = min(accs[i])
                lowers[1][i] = min(val_accs[i])
                lowers[2][i] = min(losses[i])
                lowers[3][i] = min(val_losses[i])
                print(i)
                
            if comparison ==True:
                plt.plot(list(range(self.numepochs)), np.mean(val_accs),color='red')
                plt.fill_between(list(range(self.numepochs)), lowers[1], uppers[1], facecolor='red', alpha=0.5,
                    label='10 experiments')
                #plt.set_ylim([0.0,0.8])
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                #plt.legend(["stochastic mini-batch training"])
            
            else:
                
            
                fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                ax1.plot(list(range(0,self.numepochs)), np.mean(accs))
                ax1.fill_between(list(range(0,self.numepochs)), lowers[0], uppers[0], facecolor='blue', alpha=0.5,
                       label='10 experiments')
    # =============================================================================
                ax1.plot(list(range(self.numepochs)), np.mean(val_accs))
                ax1.fill_between(list(range(self.numepochs)), lowers[1], uppers[1], facecolor='orange', alpha=0.5,
                        label='10 experiments')
                ax1.set(xlabel = "Epochs", ylabel="Accuracy")
                ax1.set_ylim([0.0,0.8])
                ax1.plot(list(range(0,self.numepochs)), np.amax(uppers[1])*np.ones(self.numepochs), color='red', linestyle='dashed')
    
                
                
                ax2.plot(list(range(self.numepochs)), np.mean(losses))
                ax2.fill_between(list(range(self.numepochs)), lowers[2], uppers[2], facecolor='blue', alpha=0.5,
                                 label='10 experiments')
                ax2.plot(list(range(self.numepochs)), np.mean(val_losses))
                ax2.fill_between(list(range(self.numepochs)), lowers[3], uppers[3], facecolor='orange', alpha=0.5,
                                 label='10 experiments')
                ax2.set(xlabel="Epochs", ylabel="Loss")
    # =============================================================================
                
    
    
                fig1.legend(['training', 'validation'])
                fig1.legend(['test'])

            #plt.show()
        


    def plotCurricResults(self, train_hist, val_hist, loss, v_loss, comparison=True):

            accs = pd.DataFrame()
            val_accs = pd.DataFrame()
            losses = pd.DataFrame()
            v_losses = pd.DataFrame()
            
            for i in range(0, len(train_hist)):
                accs = pd.concat([accs, pd.DataFrame(train_hist[i])],axis=0)
                val_accs = pd.concat([val_accs, pd.DataFrame(val_hist[i])],axis=0)
                losses = pd.concat([losses, pd.DataFrame(loss[i])],axis=0)
                v_losses = pd.concat([v_losses, pd.DataFrame(v_loss[i])],axis=0)

            if self.sto_mini_batch_comp == True:
                uppers = np.zeros((4, self.numepochs))
                lowers = np.zeros((4, self.numepochs))
                accs , val_accs = pd.DataFrame(accs), pd.DataFrame(val_accs)
                losses, v_losses = pd.DataFrame(losses), pd.DataFrame(v_losses)
                
                
                for i in range(0, self.numepochs):
                    uppers[0][i] = max(accs[i])
                    uppers[1][i] = max(val_accs[i])
                    uppers[2][i] = max(losses[i])
                    uppers[3][i] = max(v_losses[i])
    
                    lowers[0][i] = min(accs[i])
                    lowers[1][i] = min(val_accs[i])
                    lowers[2][i] = min(losses[i])
                    lowers[3][i] = min(v_losses[i])

                    print(i)
            
            else:
                uppers = np.zeros((4, self.curriculum_batches*self.curriculum_recursion))
                lowers = np.zeros((4, self.curriculum_batches*self.curriculum_recursion))
                accs , val_accs = pd.DataFrame(accs), pd.DataFrame(val_accs)
                losses, v_losses = pd.DataFrame(losses), pd.DataFrame(v_losses)
                
            
                for i in range(0, self.curriculum_batches*self.curriculum_recursion):
                    uppers[0][i] = max(accs[i])
                    uppers[1][i] = max(val_accs[i])
                    uppers[2][i] = max(losses[i])
                    uppers[3][i] = max(v_losses[i])
    
    
    
                    lowers[0][i] = min(accs[i])
                    lowers[1][i] = min(val_accs[i])
                    lowers[2][i] = min(losses[i])
                    lowers[3][i] = min(v_losses[i])

                    print(i)
            
            if comparison == True and self.sto_mini_batch_comp == True:
                plt.plot(list(range(self.numepochs)), np.mean(val_accs),color='red')
                plt.fill_between(list(range(self.numepochs)), lowers[1], uppers[1], facecolor='red', alpha=0.5, label='10 experiments')
            elif comparison == True and self.sto_mini_batch_comp != True:    
                plt.plot(list(range(self.curriculum_batches*self.curriculum_recursion)), np.mean(val_accs),color='orange')
                plt.fill_between(list(range(self.curriculum_batches*self.curriculum_recursion)), lowers[1], uppers[1], facecolor='orange', alpha=0.5, label='10 experiments')
                

                
            else:
                fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
                ax1.plot(list(range(0,self.curriculum_batches*self.curriculum_recursion)), np.mean(accs))
                ax1.fill_between(list(range(0,self.curriculum_batches*self.curriculum_recursion)), lowers[0], uppers[0], facecolor='blue', alpha=0.5,
                        label='10 experiments')
                ax1.plot(list(range(self.curriculum_batches*self.curriculum_recursion)), np.mean(val_accs))
                ax1.fill_between(list(range(self.curriculum_batches*self.curriculum_recursion)), lowers[1], uppers[1], facecolor='orange', alpha=0.5,
                        label='10 experiments')
                ax1.set(xlabel = "Number of curriculum batches", ylabel="Accuracy")
                ax1.set_ylim([0.0,0.8])
                ax1.plot(list(range(0,self.curriculum_batches*self.curriculum_recursion)), np.amax(uppers[1])*np.ones(self.curriculum_batches*self.curriculum_recursion), color='red', linestyle='dashed')
                
                ax2.plot(list(range(self.curriculum_batches*self.curriculum_recursion)), np.mean(losses))
                ax2.fill_between(list(range(self.curriculum_batches*self.curriculum_recursion)), lowers[2], uppers[2], facecolor='blue', alpha=0.5,
                        label='10 experiments')
                ax2.plot(list(range(self.curriculum_batches*self.curriculum_recursion)), np.mean(v_losses))
                ax2.fill_between(list(range(self.curriculum_batches*self.curriculum_recursion)), lowers[3], uppers[3], facecolor='orange', alpha=0.5,
                        label='10 experiments')
                ax2.set(xlabel="Number of curriculum batches", ylabel="Loss")
                
    
                
                fig1.legend(['training', 'validation'])
            
            #plt.show()

class Bandit(): 
    def __init__(self, builder_instance, model, inputs, outs, val_x, val_y): 
        self.curriculum_batches = builder_instance.curriculum_batches
        self.arm_values = np.random.normal(0,1,self.curriculum_batches) 
        self.K = np.zeros(self.curriculum_batches) 
        self.est_values = np.zeros(self.curriculum_batches)
        self.model = model
        self.inputs = inputs
        self.outs = outs
        self.val_x = val_x
        self.val_y = val_y

    def get_reward(self,action): 
        #noise = np.random.normal(0,0.1) 
        self.model.train_on_batch(self.inputs[action], self.outs[action])
        reward = self.model.test_on_batch(val_x, val_y)[1] #+ noise 
        #reward2 = (self.model.test_on_batch(train_x, train_y)[1]+self.model.test_on_batch(val_x,val_y)[1])/2
        return reward

    def choose_eps_greedy(self,epsilon):
        rand_num = np.random.random() 
        if epsilon>rand_num: 
          return np.random.randint(self.curriculum_batches) 
        else: 
          return np.argmax(self.est_values)

    def update_est(self,action,reward): 
        noise = np.random.normal(0,0.1) 
        self.K[action] += 1 
        alpha = 1./self.K[action] 
        self.est_values[action] += alpha * (reward+noise - self.est_values[action]) # keeps running average of rewards
    
    
    def experiment(self,bandit,Npulls,epsilon):
        history = [] 
        for i in range(Npulls): 
            action = bandit.choose_eps_greedy(epsilon)
            R = bandit.get_reward(action)
            if i > 0 and R > max(history):
                self.model.save('Best_bandit_model.h5')
            bandit.update_est(action,R) 
            history.append(R) 
        return np.array(history)





def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
        
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(curriculum, bandit, train_x, train_y, val_x, val_y,source):
    
    if curriculum == False:
        
        print('Batch training mode activated')
        train_hist = dict()
        val_hist = dict()
        loss = dict()
        v_loss = dict()
        hist = dict()
        for i in range(0,5):
            builder = buildNetwork(seed = i, hidden_layers = 2, hidden_nodes = 100,\
                                   dropout = 0.2, batchnorm = True, sto_mini_batch_comp=True, numepochs = 35000, batchsize=1000)
            
            if builder.sto_mini_batch_comp==True:
                model = builder.build(train_x, uniques)
                model = builder.compiler(model,source,q_net = False)
                model, training_performance, validation_performance = builder.train(model, train_x, train_y, val_x, val_y)
                builder.extractEmbedding(curriculum, bandit, model, train_x, train_y)
                training_performance.columns = list(range(0,training_performance.shape[1]))
                validation_performance.columns = list(range(0, validation_performance.shape[1]))
                train_hist[i] = training_performance[1:2]
                val_hist[i] = validation_performance[1:2]
                loss[i] = training_performance[0:1]
                v_loss[i] = validation_performance[0:1]
            
            else:
                model = builder.build(train_x, uniques)
                model = builder.compiler(model,source,q_net = False)
                model, history = builder.train(model, train_x, train_y, val_x, val_y)
                builder.extractEmbedding(curriculum, bandit, model, train_x, train_y)
                hist[i] = history
                print('Experiment ' + str(i+1) +' complete')        
            
        if builder.sto_mini_batch_comp==True:
             builder.plotCurricResults(train_hist, val_hist, loss, v_loss,comparison=True)
        else:
            save_obj(hist, 'histories')
            builder.plotResults(hist,comparison=True)
        
        
    elif curriculum == True and bandit == False:
        
        print('Curriculum only mode activated')
        train_hist = dict()
        val_hist = dict()
        loss = dict()
        v_loss = dict()
        for i in range(0,5):
            builder = buildNetwork(seed = i, hidden_layers = 2, hidden_nodes = 100, \
                                   dropout = 0.2, batchnorm = True,sto_mini_batch_comp=False,\
                                   numepochs=100,curriculum_recursion = 5000, curriculum_batches = 7)
            model = builder.build(train_x, uniques)
            model = builder.compiler(model, source, q_net=False)
            for j in range(0,1):
                model, train_perform, val_perform, batches, outs = builder.train_curriculum(model,\
                                                                                            train_x,\
                                                                                            train_y,\
                                                                                            val_x,\
                                                                                            val_y,\
                                                                                            builder.curriculum_batches,\
                                                                                            builder.curriculum_recursion)
            builder.extractEmbedding(curriculum, bandit, model, train_x, train_y)
            train_perform.columns = list(range(0,train_perform.shape[1]))
            val_perform.columns = list(range(0, val_perform.shape[1]))
            train_hist[i] = train_perform[1:2]
            val_hist[i] = val_perform[1:2]
            loss[i] = train_perform[0:1]
            v_loss[i] = val_perform[0:1]
            print('Experiment ' + str(i+1) +' complete')   
            
            print(str(i) + str(max(val_perform[1:2])))
        #save_obj(train_hist, 'train_curriculum_histories')
        #save_obj(val_hist, 'validation_curriculum_histories')
    
        builder.plotCurricResults(train_hist, val_hist, loss, v_loss,comparison=True)
        
    else:
        print('Bandit mode activated')
        train_hist = dict()
        val_hist = dict()
        loss = dict()
        v_loss = dict()
        for i in range(0,5):
            builder = buildNetwork(seed = i, hidden_layers = 2, hidden_nodes = 100, \
                                       dropout = 0.2, batchnorm = True,sto_mini_batch_comp=False,\
                                       numepochs=100,curriculum_recursion = 4000, curriculum_batches = 7, \
                                       temp = 1)
            model = builder.build(train_x, uniques)
            model = builder.compiler(model,source,q_net=False)
            for j in range(0,1):
                model, train_perform, val_perform, batches, outs = builder.train_curriculum(model,\
                                                                                            train_x,\
                                                                                            train_y,\
                                                                                            val_x,\
                                                                                            val_y,\
                                                                                            builder.curriculum_batches,\
                                                                                            builder.curriculum_recursion)        
            #builder.extractEmbedding(curriculum, bandit, model, train_x, train_y)
            train_perform.columns = list(range(0,train_perform.shape[1]))
            val_perform.columns = list(range(0, val_perform.shape[1]))
            train_hist[i] = train_perform[1:2]
            val_hist[i] = val_perform[1:2]
            loss[i] = train_perform[0:1]
            v_loss[i] = val_perform[0:1]        
            
            print('bandit begins')
            #print(val_hist.shape)
            bandito = Bandit(builder, model, batches, outs, val_x, val_y)
            #bandito = Bandit(model, batches, outs, val_x, val_y)
            Npulls = 7000
            avg_outcome_eps0p0 = np.zeros(Npulls)
            avg_outcome_eps0p0 += bandito.experiment(bandito,Npulls,0.2) 
            
            print(avg_outcome_eps0p0.shape)
            
            y = pd.DataFrame(val_perform[1:2]).T
            y.columns=[0]
            val_hist[i] = pd.concat([y, pd.DataFrame(avg_outcome_eps0p0)],axis=0, ignore_index=True)
        
        
        uppers = np.zeros((2, val_hist[0].shape[0]))
        lowers = np.zeros((2, val_hist[0].shape[0]))

        overall_valhist = pd.concat([val_hist[0].T, val_hist[1].T], axis=0)
        overall_valhist = pd.concat([overall_valhist, val_hist[2].T], axis=0)
        overall_valhist = pd.concat([overall_valhist, val_hist[3].T], axis=0)
        overall_valhist= pd.concat([overall_valhist, val_hist[4].T], axis=0)
        
        mean = np.mean(overall_valhist)

        for i in range(0, val_hist[0].shape[0]):
            #uppers[0][i] = max(overall_valhist[i])
            uppers[1][i] = max(overall_valhist[i])
            #uppers[2][i] = max(loss[i])
            #uppers[3][i] = max(v_loss[i])



            #lowers[0][i] = min(train_hist[i])
            lowers[1][i] = min(overall_valhist[i])
            #lowers[2][i] = min(loss[i])
            #lowers[3][i] = min(v_loss[i])

        #fig1 = plt.figure(3)
        #ax1 = fig1.gca()
        plt.plot(list(range(0,overall_valhist.shape[1])), mean, color='green')
        plt.fill_between(list(range(0,overall_valhist.shape[1])), lowers[1], uppers[1], facecolor='green', alpha=0.5,
                    label='10 experiments')
        #ax1.set(xlabel="Number of curriculum batches", ylabel="Accuracy")
        #ax1.set_ylim([0.0,1.0])
        plt.plot(list(range(0,overall_valhist.shape[1])), np.amax(uppers[1])*np.ones(overall_valhist.shape[1]), color='red', linestyle='dashed')
        plt.legend(["stochastic mini-batches", "curriculum", "curriculum + MAB"])
        plt.xlabel('Number of batches')
        plt.ylabel('Test Accuracy')
        #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=str2bool, help='Do you want to compare all methods?')
    parser.add_argument("--source", type = str, help="Specify source of data")
    parser.add_argument("--red_feat", type=str2bool, help='Reduce features to salient only?')
    parser.add_argument("--balanced_class", type=str2bool, help='Balance classes?')
    parser.add_argument("--greyout", type=str2bool, help='dataprocessing')
    parser.add_argument("--curriculum", type = str2bool, help="use a curriculum for training or not")
    parser.add_argument("--bandit", type = str2bool, help="use multi-armed bandit for training or not")
    args = parser.parse_args()
    source = args.source
    red_feat = args.red_feat
    balanced_class = args.balanced_class
    greyout = args.greyout
    curriculum = args.curriculum    
    bandit = args.bandit
    comparison = args.comparison
    
 
    
    if source == "PAMAP2" or source == 'news' or source == 'frogs' or source =='drive' or source=='diabetes':
        from PAMAP2_script import PAMAP2_data
        pamd = PAMAP2_data(source)
        train_x, train_y, val_x, val_y, test_x, test_y, uniques = pamd.getTrainingData()
    else:
        gtd = generateTrainingData(source=source,red_feat=red_feat,balanced_class=balanced_class,greyout=greyout)
        train_x, train_y, val_x, val_y, test_x, test_y, uniques = gtd.getDataSplit()
        encoder, history = gtd.encodedInputs(train_x, val_x)
        train_x = pd.DataFrame(encoder.predict(train_x))
        val_x = pd.DataFrame(encoder.predict(val_x))
        test_x = pd.DataFrame(encoder.predict(test_x))
#     
# =============================================================================
    if comparison == True:
        gtd = generateTrainingData(source=source,red_feat=True,balanced_class=False,greyout=True)
        train_x, train_y, val_x, val_y, test_x, test_y, uniques = gtd.getDataSplit()
        encoder, history = gtd.encodedInputs(train_x, val_x)
        train_x = pd.DataFrame(encoder.predict(train_x))
        val_x = pd.DataFrame(encoder.predict(val_x))
        test_x = pd.DataFrame(encoder.predict(test_x))
        
        main(False, False, train_x, train_y, val_x, val_y,source)
        
# =============================================================================
#         gtd = generateTrainingData(source=source,red_feat=False,balanced_class=False,greyout=False)
#         train_x, train_y, val_x, val_y, test_x, test_y, uniques = gtd.getDataSplit()
#         encoder, history = gtd.encodedInputs(train_x, val_x)
#         train_x = pd.DataFrame(encoder.predict(train_x))
#         val_x = pd.DataFrame(encoder.predict(val_x))
#         test_x = pd.DataFrame(encoder.predict(test_x))
# =============================================================================
        
        main(True, False, train_x, train_y, val_x, val_y,source)
        main(True, True, train_x, train_y, val_x, val_y,source)
    else:
        main(curriculum, bandit, train_x, train_y, val_x, val_y,source)
    
    plt.show()


    
            

