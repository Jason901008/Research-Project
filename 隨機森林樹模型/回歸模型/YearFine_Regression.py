# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:02:51 2024

@author: user
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from text2vec import SentenceModel

def input_data():
    global train_data
    global test_data
    
    #train data
    train_data = pd.read_csv('./Train&Test_EvaluateData/Period_train/Negligent_Injury_ner_for_WordEbedding_Period_train.csv', encoding='utf-8-sig', header=None)
    train_data.columns = ['keywords', 'years', 'fine', 'name']
    
    #test data
    test_data = pd.read_csv('./Train&Test_EvaluateData/Period_test/Negligent_Injury_ner_for_WordEbedding_Period_test.csv', encoding='utf-8-sig', header=None)
    test_data.columns = ['keywords', 'years', 'fine', 'name']

def transfer_data():
    global X_train_embedding , X_test_embedding
    global y_train_years , y_test_years
    global y_train_fine , y_test_fine
    
    X_train = train_data['keywords']
    X_test  = test_data['keywords']

    m = SentenceModel("shibing624/text2vec-base-multilingual")
    X_train_embedding = m.encode(X_train)
    X_test_embedding = m.encode(X_test.apply(lambda x: np.str_(x)))
    
    y_train_years = train_data['years'].astype(float)
    y_test_years  = test_data['years'].astype(float)
    
    y_train_fine = train_data['fine'].astype(int)
    y_test_fine  = test_data['fine'].astype(int)
    
def training():
    global y_pred_years , y_pred_fine
    
    #train & predict 'years'
    rf_regressor_years = RandomForestRegressor()
    rf_regressor_years.fit(X_train_embedding, y_train_years)
    y_pred_years = rf_regressor_years.predict(X_test_embedding)

    #train & predict 'fine'
    rf_regressor_fine = RandomForestRegressor()
    rf_regressor_fine.fit(X_train_embedding, y_train_fine)
    y_pred_fine = rf_regressor_fine.predict(X_test_embedding)

#------------------------------------------------------------------------------
input_data()
transfer_data()
training()




