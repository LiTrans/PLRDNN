
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from keras.models import  Model,Input
from keras.layers import Dense,Activation,Dropout,BatchNormalization
from keras import Input
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import cross_val_score
from math import ceil
from keras.wrappers.scikit_learn import KerasClassifier
import time
from sklearn.utils import shuffle        #PseudoLabeler definition
from sklearn.base import BaseEstimator, ClassifierMixin




def Unit(x):
    res = x
    out = BatchNormalization()(x)
    out=Activation("relu")(out) 
    out=Dense(10)(out)

    out = BatchNormalization()(out)
    out=Activation("relu")(out) 
    out=Dense(10)(out)
    out = keras.layers.add([res,out])
    return out

def MiniModel(hU=2):   #number of hidden units

    input_shape=(15,)
    
    images=Input(input_shape)
    net=Dense(10, activation='relu')(images)

    for i in range(hU):
        net = Unit(net)
        
   


    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dense(units=3,activation="softmax")(net)

    model = Model(inputs=images,outputs=net)
    model.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

class PseudoLabeler(BaseEstimator, ClassifierMixin):
    '''
    Sci-kit learn wrapper for creating pseudo-lebeled estimators.
    '''

    def __init__(self, model, unlabled_data, features, target, sample_rate=0.2, seed=42):
        '''
        @sample_rate - percent of samples used as pseudo-labelled data
                       from the unlabled dataset
        '''
        assert sample_rate <= 1.0, 'Sample_rate should be between 0.0 and 1.0.'

        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed

        self.unlabled_data = unlabled_data
        self.features = features
        self.target = target

    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "unlabled_data": self.unlabled_data,
            "features": self.features,
            "target": self.target
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


    def fit(self, X, y):
        '''
        Fit the data using pseudo labeling.
        '''
        augemented_train = self.__create_augmented_train(X, y)
        self.model.fit(
            augemented_train[:,0:15],
            augemented_train[:,15]
        )

        return self


    def __create_augmented_train(self, X, y):
        '''
        Create and return the augmented_train set that consists
        of pseudo-labeled and labeled data.
        '''        
        num_of_samples = int(len(self.unlabled_data) * self.sample_rate)

        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.unlabled_data[self.features])

        # Add the pseudo-labels to the test set
        pseudo_data = self.unlabled_data.copy(deep=True)
        pseudo_data[self.target] = pseudo_labels

        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_pseudo_data = pseudo_data.sample(n=num_of_samples)
        c, r = sampled_pseudo_data.shape
        d= y.shape[0]
       
        sampled_pseudo_data=sampled_pseudo_data.values.reshape(c,16)
        temp_train = np.concatenate([X, y.reshape(d,1)], axis=1)
        augemented_train = np.concatenate([sampled_pseudo_data, temp_train])

        return shuffle(augemented_train)

    def predict(self, X):
        '''
        Returns the predicted values.
        '''
        return self.model.predict(X)

    def get_model_name(self):
        return self.model.__class__.__name__

