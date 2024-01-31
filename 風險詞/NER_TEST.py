import pickle
import pandas as pd
from ckip_transformers.nlp import CkipNerChunker


#ner_driver_4 = CkipNerChunker ( model_name = "D:/Downloads/MODEL",device = 0 )
ner_driver_4 = CkipNerChunker ( model_name = r"../../MODEL",device = 0 )

csv_file_path = r"Result_Record/Drug_original_data.csv"

df = pd.read_csv ( csv_file_path , header=None )
test_data = df[0].tolist()
true_answers = df[1].tolist()

ner_4 = ner_driver_4 ( test_data )


# Write & Read NER File
with open ( r'Drug_predict_ner.pkl', 'wb' ) as ner_file :
    pickle.dump ( ner_4 , ner_file )

with open( r'test_predict_ner.pkl', 'rb') as ner_file:
    ner_4_read = pickle.load ( ner_file )
