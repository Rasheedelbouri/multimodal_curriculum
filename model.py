#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:41:08 2020

@author: kebl4170
"""
import numpy as np
from keras.layers import Activation, Conv2D, MaxPooling2D, Dense, BatchNormalization, LSTM, Dropout, Input, Conv1D, Flatten
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
import keras.backend as K
import tensorflow as tf
import keras
    

class buildNetwork():
    
    def __init__(self, source='H', seed = 0, hidden_layers=2, hidden_nodes=50, temp=1, dropout=0.2,\
                 activation='relu', batchnorm=True, numepochs=100, batchsize=30 \
                 ,curriculum_batches = 10, curriculum_recursion = 1, q_net = False,
                 act_net = False, crit_net=False):
        self.source = source
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
        self.q_net = q_net
        self.act_net = act_net
        self.crit_net = crit_net
        

    def build(self, train_x, uniques,conv=True,generative=False,continual=False):
        np.random.seed(self.seed) # setting initial seed for reproducable results
        temp = self.temp # setting temperature of softmax functipn
        self.generative = generative
        self.continual = continual
        
        if self.source == 'cifar':
            model = Sequential()
            model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=train_x.shape[1:]))
            model.add(Activation('relu'))
            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
    
    
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
    
            model.add(Flatten())
            model.add(Dense(100))
            model.add(Activation('relu'))
            model.add(Dropout(0.1))
            model.add(Dense(70))
            model.add(Activation('relu'))
            model.add(Dropout(0.1))
            model.add(Dense(70))
            model.add(Activation('relu'))
            model.add(Dropout(0.1))
            model.add(Dense(len(uniques)))
            model.add(Activation('softmax'))

            return model

        if conv == True:
            inputs = Input(shape = (train_x.shape[1],1))

            conv = Conv1D(4,7,strides=3,activation='relu',input_shape=(None,train_x.shape[1],1))(inputs)
            #maxp = MaxPool1D()(conv)
            conv2 = Conv1D(8,7,strides=3,activation='relu')(conv)
            #maxp2 = MaxPool1D()(conv2)
            conv3 = Conv1D(16,7,strides=3,activation='relu')(conv2)
            #maxp3 = MaxPool1D()(conv3)
            
            #lstm = LSTM(self.hidden_nodes)(conv3)
            #lstm2 = LSTM(self.hidden_nodes)(lstm)
            flat = Flatten()(conv3)
            
            
        #tf.reshape(conv2, (conv2.shape[1]*conv2.shape[2]))
        
            hidden1 = Dense(self.hidden_nodes, activation = self.activation)(flat)
        else:
            if self.generative == True:
                if self.source == 'M':
                    inputs = Input(shape = [train_x.shape[1] + len(uniques)-1])
                else:
                    inputs = Input(shape = [train_x.shape[1] + len(uniques)])
            else:
                inputs = Input(shape = [train_x.shape[1]])
                
            hidden1 = Dense(self.hidden_nodes, activation = self.activation)(inputs)
        do1 = Dropout(self.dropout)(hidden1)
        if self.batchnorm == True:
            bn1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(do1)
            hidden2 = Dense(self.hidden_nodes, activation = self.activation)(bn1)
        else:
            hidden2 = Dense(self.hidden_nodes, activation = self.activation)(do1)

        do2 = Dropout(self.dropout)(hidden2)
        
        
        if self.batchnorm == True:
            bn3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(do2)
            hidden3 = Dense(self.hidden_nodes, activation = self.activation)(bn3)
        else:
            hidden3 = Dense(self.hidden_nodes, activation = self.activation)(do2)
            
        do3 = Dropout(self.dropout)(hidden3)
        
        
        if self.source == 'M':
            if self.generative == True:
                out = Dense(train_x.shape[1], activation=None)(do3)
                model = Model(inputs=inputs, outputs=out)
            elif self.continual == True:
                branchA = Dense(4, activation='softmax')(do3)
                branchB = Dense(1, activation='sigmoid')(do3)
                branchC = Dense(1, activation='sigmoid')(do3)
                out = concatenate([branchA, branchB, branchC])
                
                model = Model(inputs=inputs, outputs=out)
            else:
                out = Dense(1, activation='sigmoid')(do3)
                model = Model(inputs=inputs, outputs=out)
        else:
            if self.act_net == True:
                if self.batchnorm == True:
                    bn2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(do3)
                    branchA = Dense(1, activation = "tanh")(bn2)
                    branchB = Dense(1, activation = "sigmoid")(bn2)
                    branchC = Dense(1, activation = "sigmoid")(bn2)
        
                    out = concatenate([branchA, branchB, branchC])
                
                    model = Model(inputs=inputs, outputs=out)
                else:
                    branchA = Dense(1, activation = "tanh")(do3)
                    branchB = Dense(1, activation = "sigmoid")(do3)
                    branchC = Dense(1, activation = "sigmoid")(do3)
                    out = concatenate([branchA, branchB, branchC])
                    model = Model(inputs=inputs, outputs=out)

            elif self.crit_net == True:
                if self.batchnorm == True:
                    bn2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(do3)
                    out = Dense(3, activation = None)(bn2)
                    model = Model(inputs=inputs, outputs=out)
                else:
                    out = Dense(3, activation = None)(do3)
                    model = Model(inputs=inputs, outputs=out)
            else:
                if self.source != 'M':
                    if self.generative == True:
                        out = Dense(train_x.shape[1], activation=None)(do3)
                        model = Model(inputs=inputs, outputs=out)
                    elif self.continual == True:
                        branchA = Dense(6, activation='softmax')(do3)
                        branchB = Dense(1, activation='sigmoid')(do3)
                        branchC = Dense(1, activation='sigmoid')(do3)

                        out = concatenate([branchA, branchB, branchC])
                        
                        model = Model(inputs=inputs, outputs=out)
                    else:
                        out = Dense(len(uniques), activation='softmax')(do3)
                        model = Model(inputs=inputs, outputs=out)
                else:
                    out = Dense(len(uniques)-1, activation='sigmoid')(do3)
                    model = Model(inputs=inputs, outputs=out)
        
        
        return(model)
        
    def customLoss(self, yTrue, yPred):
        
        #yPred=0
        loss = tf.reduce_mean(0*yPred + yTrue)
        return (loss)
    
    
    def compiler(self, model, q_net, actor):
        
        if self.source == 'cifar':
            opt = keras.optimizers.Adam()
            model.compile(loss='categorical_crossentropy', optimizer=opt,
                          metrics=['accuracy'])
            
            return model
        
        sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        if self.generative == True:
            model.compile(loss=self.customLoss,
                              optimizer='adam',
                              metrics=['accuracy'])
            
        if q_net == True:
            if actor == True:
                model.compile(loss=self.customLoss,
                              optimizer='adam',
                              metrics=['accuracy'])
            else:  
                model.compile(loss='mean_squared_error',# self.customLoss, #self.my_loss, 
                      optimizer= 'adam', 
                      metrics=['accuracy'])
        else:
            if self.source == 'M':
                sgd = SGD(lr = 0.01)
                model.compile(loss='binary_crossentropy', #self.my_loss, 
                      optimizer= sgd, 
                      metrics=['accuracy'])
            else:
                #sgd = SGD(lr = 0.0001,momentum=0.9)
                model.compile(loss='categorical_crossentropy', #self.my_loss, 
                          optimizer= sgd,
                          metrics=['accuracy'])
            
        return(model)