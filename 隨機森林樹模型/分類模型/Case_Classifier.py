# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:41:03 2024

@author: user
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from text2vec import SentenceModel


def input_train_data():
    global merged_df
    
    #train data
    folder = './Train&Test_EvaluateData/Period_train'
    paths = [os.path.join(folder, filename) for filename in os.listdir(folder)]

    dfs = []
    for path in paths:
        df = pd.read_csv(path, encoding='utf-8-sig', header=None)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    
def tranfer_label():    
    global instance
    
    #transfer case name into number
    label_encoder = LabelEncoder()
    instance  = ['毒品危害防制條例' , '偽造文書' , '賭博' , '過失傷害' , '公共危險' , '竊盜']
    English_instance = ['Drug','Forged_Documents','Gamble','Negligent_Injury','Public_Danger','Theft']
    case_conditions = {}
    for i in range (len(instance)):
        case_conditions[f'case{i+1}'] = (merged_df[3] == instance[i])
    
    for case_name, condition in case_conditions.items():
        merged_df[f'label_{case_name}'] = label_encoder.fit_transform(condition)    
      
def input_test_data():
    global dfs_test_all
    
    #test data 
    test_folder = './Train&Test_EvaluateData/Period_test'
    test_paths = [os.path.join(test_folder, filename) for filename in os.listdir(test_folder)]

    dfs_test = []
    for path in test_paths:
        df_test = pd.read_csv(path, encoding='utf-8-sig', header=None)
        dfs_test.append(df_test)
          
    dfs_test_all = pd.concat(dfs_test, ignore_index=True)

def training():
    global test_embedding
    global case
    
    #Transfer train & test data into Embedding
    m = SentenceModel("shibing624/text2vec-base-multilingual")
    train_embedding = m.encode(merged_df[0])
    test_embedding = m.encode(dfs_test_all[0].apply(lambda x: np.str_(x)))
    
    #Use Randomforest to train classifier model
    rf_classifier_seg_drug = RandomForestClassifier()
    rf_classifier_seg_forged_documents = RandomForestClassifier()
    rf_classifier_seg_gamble = RandomForestClassifier()
    rf_classifier_seg_negligent_injury = RandomForestClassifier()
    rf_classifier_seg_public = RandomForestClassifier()
    rf_classifier_seg_theft = RandomForestClassifier()

    rf_classifier_seg_drug.fit(train_embedding, merged_df['label_case1']==1)  
    rf_classifier_seg_forged_documents.fit(train_embedding, merged_df['label_case2']==1)  
    rf_classifier_seg_gamble.fit(train_embedding, merged_df['label_case3']==1)
    rf_classifier_seg_negligent_injury.fit(train_embedding, merged_df['label_case4']==1)  
    rf_classifier_seg_public.fit(train_embedding, merged_df['label_case5']==1)  
    rf_classifier_seg_theft.fit(train_embedding, merged_df['label_case6']==1)
    
    case = [rf_classifier_seg_drug, rf_classifier_seg_forged_documents, rf_classifier_seg_gamble, rf_classifier_seg_negligent_injury, rf_classifier_seg_public ,rf_classifier_seg_theft]
    
def Find_case():
    global max_proba_case 
    
    max_proba_index = []
    max_proba_case = []

    for i in range(len(dfs_test_all)):
        probs_all_list = []
        probs_list = []
        for j in range(len(case)):
            test = np.array(test_embedding[i]).reshape(1 , -1)
            probs = case[j].predict_proba(test)
            probs_all_list.append(probs)
        
            #choose the case which is yes
            probs_list.append(probs[0][1])
            
        #choose the biggest Probability
        max_proba = max(probs_list)
        max_proba_index.append(probs_list.index(max_proba))
        max_proba_case.append(instance[probs_list.index(max_proba)])
    

input_train_data()
tranfer_label()
input_test_data()
training()
Find_case() 



