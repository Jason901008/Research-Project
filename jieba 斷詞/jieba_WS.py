import jieba
import pandas as pd



pred_risk_file = r'Result_Record/Public_Danger/ner/Public_Danger_interface_Pred_all_risk.csv'
user_dict_file = r'Ws_Dictionary.txt'



# Read Predict Risk
df = pd.read_csv ( pred_risk_file , header = None )
pred_risk = df[0].tolist()
for i in range ( (len(pred_risk) ) ) :
    if type( pred_risk[i] ) == float :
        pred_risk[i] = ""

# User Dictionary
with open ( user_dict_file , 'r' , encoding = 'utf-8' ) as f :  
    jieba.load_userdict ( f )                



# Span Word Original_Data & Predict Risk
def Word_Span () :
    global result
    global pred_WS
    
    pred_WS = []
    
    # Interface Risk
    for i in range ( len(pred_risk) ) :
        result = ""
        seg_list = jieba.cut ( pred_risk[i], cut_all=False, HMM=True )
        #for w in seg_list if word not in stop_words                        #去掉停用詞
        for w in seg_list :
            result += w + " "
            #ws.append ( w )    
        
        pred_WS.append ( result )



Word_Span ()




























