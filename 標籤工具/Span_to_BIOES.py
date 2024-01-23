import json
import pandas as pd

#input_file = r'Ner_data/01/All_1.json'
#input_file = r'All_Risk_5.json'
input_file = r'Result_Record/Theft/Bonus/Theft_400.json'

with open ( input_file , 'r' , encoding = 'utf-8' ) as f :
    data = json.load(f)
    


# Span to BIOES
def to_BIOES () : 
    global all_data
    global all_text
    global all_verdict_data
    global all_verdict_risk
    
    all_data = []
    all_text = {}
    all_verdict_data = []
    all_verdict_risk = []
    
    
    for verdict in range ( 0 , len ( data ) ) :
        item = data [ verdict ]
        spans = item['annotations'][0]['result']
        tokens = item['data']['text']
        spilt_verdict = [ char for char in tokens ]                 # Convert text into a list of characters
        
        line = ""
        bio_labels = ['O'] * len(spilt_verdict)
        
        for span in spans :
            #print ( verdict , span)
            start = span['value']['start']
            end = span['value']['end']
            entity_type = span['value']['labels'][0]
            text = span['value']['text']
            
            dis = end-start
            line = line + " " + text
            
            # Transform
            if dis == 1 :
                bio_labels[start] = 'S-' + entity_type
                print ( text , verdict , verdict%510 )
            else :
                bio_labels[start] = 'B-' + entity_type
                for i in range(start + 1, end):
                    bio_labels[i] = 'I-' + entity_type
                bio_labels[end-1] = 'E-' + entity_type
                
            # Record Text
            if text not in all_text.keys() :
                all_text [ text ] = 1
            else :
                all_text [ text ] += 1
            
        d = {'tokens':spilt_verdict,'tags':bio_labels}
        
        # Push Verdict BIOES
        all_data.append ( d )
        all_verdict_data.append ( tokens )
        all_verdict_risk.append ( line )


# Output json
def Write_to_Json () : 
    # 檔名不管大小寫，會是同一個檔案
    #with open ( 'all_5.json' , 'w' , encoding='utf-8' ) as f :
    with open ( 'Theft_Bonus.json' , 'w' , encoding='utf-8' ) as f :
        for data in all_data :
            # 将字典数据转换为没有额外空白字符的 JSON 字符串
            json_data = json.dumps ( data , separators = (',', ':') , ensure_ascii=False )
            f.write ( json_data + '\n' )



# Output txt
def Write_to_Txt () :
    #with open ( 'all_text_5.txt' , 'w' , encoding='utf-8' ) as f :
    with open ( 'Theft_Bonus.txt' , 'w' , encoding='utf-8' ) as f :
        for text in all_text.keys() :
            f.write ( text + '\n' )



# Output All Verdict Original Data
def Write_to_Csv () :
    global all_df
    
    all_data = { 'Text': all_verdict_data , 'Risk': all_verdict_risk }
    all_df = pd.DataFrame ( all_data )
    #all_df.to_csv ( 'Risk_all_data.csv', index=False, header=0, encoding='utf-8' )
    all_df.to_csv ( 'Theft_Bonus_data.csv', index=False, header=0, encoding='utf-8' )




to_BIOES ()
Write_to_Json ()
Write_to_Txt ()
Write_to_Csv ()