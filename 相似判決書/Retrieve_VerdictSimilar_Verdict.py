# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 16:18:00 2023

@author: User
"""
import os
import jieba
import numpy as np
import pandas as pd
from text2vec import SentenceModel
from text2vec import semantic_search
from ckip_transformers.nlp import CkipNerChunker
from sklearn.feature_extraction.text import TfidfVectorizer


verdict_file = r'sim_data/訓練資料/Drug/Drug_original_data.csv'
all_result_file = r'sim_data/訓練資料/Drug/Drug_result.csv'
val_num_file = r'sim_data/訓練資料/Drug/Drug_val_number.csv'
ner_file = r'sim_data/訓練資料/Drug/Drug_predict_ner.pkl'


# Read Original Data & Risk
df = pd.read_csv ( verdict_file , header = None )
original_data = df[0].tolist()
original_risk = df[1].tolist()

# Read Record Result
record_result = pd.read_csv ( all_result_file , header = None )

# Read Valudation Number
df = pd.read_csv ( val_num_file , header = None )
val_num = df[0].tolist()

# Testing test data
csv_file_path = r"sim_data/test_data.csv"

file = pd.read_csv(csv_file_path ,encoding = 'utf-8', header=None)
data = file[0].tolist()

ner_model = CkipNerChunker ( model_name=r"../../MODEL" )
ner_4_read = ner_model(data)

#------------------------------------------------------------------------------
#The section of context

# Show NER
def show_ner () :
    global pred_result
    global pred_idx
    
    pred_result = []
    pred_idx = []
    
    for sentence_ner_4 in ner_4_read :
       pred_ans = ""
       idx = []
       
       for entity in sentence_ner_4: 
          #print(entity)
          pred_ans = pred_ans + " " +entity.word
          idx.append ( entity.idx )
       
       pred_result.append ( pred_ans )
       pred_idx.append ( idx )


# Find punctuation position in verdict
def find_punctuation () :
    global punct_pos
    
    punct_pos = []
    punctuation = [ '，' , '；' , '。' ]
    
    for verdict in data :
        i = 0
        pos = []
        
        for word in verdict :
            if word in punctuation :
                pos.append ( i )
            i += 1
        
        punct_pos.append ( pos )


# Mark risk sentences
def mark_risk_sentences () :
    global risk_sentence_pos

    risk_sentence_pos = []    # N : not need, R : risk sentences, C : context

    
    for i in range ( len ( data ) ) :
        risk_sentence = [ 'N' ] * len ( punct_pos [ i ] )
        
        for idx in pred_idx [ i ] :
            pre_pos = 0
            for j in range ( len ( punct_pos [ i ] ) ) :
                if pre_pos < idx [ 0 ] and idx [ 0 ] < punct_pos[i] [ j ] :
                    #print ( test_data_punct_pos[i] [ j ] , ' - ' , idx )
                    risk_sentence [ j ] = 'R'
                pre_pos = punct_pos[i] [ j ]
        
        risk_sentence_pos.append ( risk_sentence )


# Mark context sentences
def mark_context_sentences () :
    dis = 1
    
    for verdict in risk_sentence_pos :
        for i in range ( len ( verdict ) ) :
            if verdict [ i ] == 'R' :
                for j in range ( -1*dis + i , dis+1 + i ) :                         # -1 <= j < 2 : range is ( -1 ~ 1 )
                   if j > -1 and j < len(verdict) and verdict [ j ] == 'N' :
                        verdict [ j ] = 'C'


# Retrieve sentences position
def retrieve_sentences_position () :
    global retrieve_sentences_pos
    
    retrieve_sentences_pos = []
    
    for i in range ( len ( punct_pos ) ) :
        senteneces_pos = []
        sentence_start_pos = 0
        
        for j in range ( len ( punct_pos [ i ] ) ) :
            if risk_sentence_pos[i] [ j ] != 'N' :
                senteneces_pos.append ( ( sentence_start_pos , punct_pos[i] [ j ] ) )
            sentence_start_pos = punct_pos[i] [ j ] + 1
        
        retrieve_sentences_pos.append ( senteneces_pos )


# Retrieve sentences
def retrieve_sentences () :
    global retrieve_verdict
    global retrieve_verdict_df
    
    retrieve_verdict = []
    
    for i in range ( len ( data ) ) : 
        verdict = ""
        flag = False
        
        for j in range ( len ( retrieve_sentences_pos [ i ] ) ) : 
            if j == len ( retrieve_sentences_pos [ i ] )-1 :
                flag = True
            elif retrieve_sentences_pos[i][j] [1]+1 != retrieve_sentences_pos[i][j+1] [0] :
                flag = True
            else :
                flag = False
            
            for p in range ( retrieve_sentences_pos[i][j] [0] , retrieve_sentences_pos[i][j] [1]+1 ) :
                if flag and p == retrieve_sentences_pos[i][j] [1] :
                    verdict += '。'
                else :
                    verdict += data[i] [ p ]
            
        retrieve_verdict.append ( verdict )
    
    retrieve_verdict_df = { 'verdict': retrieve_verdict }
    retrieve_verdict_df = pd.DataFrame ( retrieve_verdict_df )


# Merge & Cut train and test
def merge () :
    global record_all_df
    global record_train_df
    global record_test_df
    
    record_test_df = pd.DataFrame ()
    record_train_df = pd.DataFrame ()
    
    record_all_df = pd.concat( [ retrieve_verdict_df , record_result[1] , record_result[2] , record_result[3] ] , axis=1 , ignore_index=True )
    
    for i in range ( len ( original_data ) ) :
        if i in val_num :
            record_test_df = pd.concat( [ record_test_df , record_all_df.iloc [ [ i ] ] ] , axis=0 , ignore_index=True )
            
        else :
            record_train_df = pd.concat( [ record_train_df , record_all_df.iloc [ [ i ] ] ] , axis=0 , ignore_index=True )


# Output
def output () :
    record_all_df.to_csv ( 'Theft_ner_for_WordEbedding_Period_all.csv', index=False, header=0 ,encoding='utf-8' )
    record_train_df.to_csv ( 'Theft_ner_for_WordEbedding_Period_train.csv', index=False, header=0 ,encoding='utf-8' )
    record_test_df.to_csv ( 'Theft_ner_for_WordEbedding_Period_test.csv', index=False, header=0 ,encoding='utf-8' )

#------------------------------------------------------------------------------
#The section of similar verdict   

#use TF_IDF to find similar verdict  
def tfidf(classifier , number):
    # Read original data
    instance = [ 'Drug' , 'Forged_Documents' , 'Gamble' , 'Negligent_Injury' , 'Public_Danger' , 'Theft' ]
    sim_list = []
    
    for i in instance :
        path = f'sim_data/訓練資料/{i}/{i}_original_data.csv'
        df_sim = pd.read_csv( path , encoding='utf-8-sig', header=None)
        sim_list.append(df_sim[0])
    
    
    # Read train data
    folder = r'sim_data/interface_train'
    paths = [os.path.join(folder, filename) for filename in os.listdir(folder)]

    dfs = []
    for path in paths:
        df = pd.read_csv(path, encoding='utf-8-sig', header=None)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    
    
    # Change the case of verdict by user
    data_for_max_proba_case = merged_df[merged_df[3] == "竊盜"]
    
    
    #Jieba Span Word
    ner_results = []
    for i in range(len(ner_4_read[0])):
        ner_results.append(ner_4_read[0][i].word)

    segment_result = []

    segmented_text = ""
    for k in range(len(ner_results)):
        result = ""
        seg_list = jieba.cut(ner_results[k], cut_all=False, HMM=True)
        for w in seg_list:
            result += w + " "
            segmented_text += result
    segment_result.append(segmented_text)
    
    
    # Tf-Idf similar data
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(data_for_max_proba_case[0])
    X_test_tfidf = tfidf_vectorizer.transform(segment_result)

    X_train_tfidf_cos = X_train_tfidf.toarray()
    X_test_tfidf_cos = X_test_tfidf.toarray()

    b = X_test_tfidf_cos
    result_cos = []
        
    for j in range(len(X_train_tfidf_cos)):
        a = X_train_tfidf_cos[j]
        c = (a * b).T
        result_cos.append(sum(c))

    #取前N筆
    N = 10
    top_N_indices = np.argsort(result_cos, axis=0)[-N:][::-1]

    #取前N筆的verdict
    top_N_data_for_case = []
    #top_N_data = []

    for index in top_N_indices:
        if index < len(data_for_max_proba_case):
            top_N_data_for_case.append(sim_list[5][index])
        else:
            top_N_data_for_case.append(None)
            
    tfidf_text = []
    for text in top_N_data_for_case:
        tfidf_text.append(text.values[0])
        
    tfidf_text_df = pd.DataFrame(tfidf_text)
    
    
    # Output similar verdict
    tfidf_text_df.to_csv ( f'{classifier}_word_sim_tf_{number}.csv', index=False, header=False , encoding='utf-8' )


#use Embedding to find similar verdict  
def embedding(classifier , number):
    # Read original data
    instance = [ 'Drug' , 'Forged_Documents' , 'Gamble' , 'Negligent_Injury' , 'Public_Danger' , 'Theft' ]
    sim_list = []
    
    for i in instance :
        path = f'sim_data/訓練資料/{i}/{i}_original_data.csv'
        df_sim = pd.read_csv( path , encoding='utf-8-sig', header=None)
        sim_list.append(df_sim[0])
    
    # Word Embedding similar verdict
    m = SentenceModel("shibing624/text2vec-base-multilingual")
    corpus = sim_list[5]

    corpus_embeddings = m.encode(corpus)
    dfs = []

    for query in retrieve_verdict:
        query_embedding = m.encode(query)
        hits = semantic_search(query_embedding, corpus_embeddings)
        nhit = hits[0]
        for hit in nhit:
            df = corpus[hit['corpus_id']]
            dfs.append(df)

    word_df = pd.DataFrame(dfs)
    
    
    # Output similar verdict
    word_df.to_csv ( f'{classifier}_word_sim_embedding_{number}.csv', index=False, encoding='utf-8' ) 

#------------------------------------------------------------------------------
# Change classifier and number can alter file name
# Need user check the case of verdict
classifier = "theft"
number = 5

show_ner ()
find_punctuation ()
mark_risk_sentences ()
mark_context_sentences ()
retrieve_sentences_position ()
retrieve_sentences ()
merge ()
#output ()
tfidf(classifier , number)
embedding(classifier , number) 









