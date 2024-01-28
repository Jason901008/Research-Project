# 深度學習在法院判決書上的重要訊息擷取與預測之研究
### 系統介面：[法律預測系統]()
## 系統下載：[]()

* 資料來源<br>
刑事判決書來源：[司法院資料開放平台](https://opendata.judicial.gov.tw/dataset/detail?datasetId=27959)<br>
開放平台使用說明：[Source](https://github.com/Jason901008/Research-Project/blob/main/%E8%B3%87%E6%96%99%E4%BE%86%E6%BA%90/Source.pdf)<br>

* 標籤工具<br>
[label studio下載與使用說明](https://blog.csdn.net/qq_44193969/article/details/123298406)<br>
轉成BIOES格式：[Code](https://github.com/Jason901008/Research-Project/blob/main/%E6%A8%99%E7%B1%A4%E5%B7%A5%E5%85%B7/Span_to_BIOES.py)<br>

* 風險詞<br>
需要先下載 [transformers](https://github.com/huggingface/transformers/tree/main/examples)<br>
微調NER模型：[Fine-tune NER Model](https://github.com/Jason901008/Research-Project/blob/main/%E9%A2%A8%E9%9A%AA%E8%A9%9E/Fine-tune_NER_Model.pdf)<br>
雲端 [NER MODEL](https://drive.google.com/drive/u/0/folders/1Th6UCs6kKGzA38C7cvFBtrE40LQUy3Qe)<br>
判決書原文資料：[Original Data](https://github.com/Jason901008/Research-Project/tree/main/%E9%A2%A8%E9%9A%AA%E8%A9%9E/Result_Record/Original_Data)<br>
NER模型使用：[Code](https://github.com/Jason901008/Research-Project/blob/main/%E9%A2%A8%E9%9A%AA%E8%A9%9E/NER_TEST.py)<br>

* Jieba斷詞<br>
使用上述 NER 模型的 [輸出結果](https://github.com/Jason901008/Research-Project/tree/main/jieba%20%E6%96%B7%E8%A9%9E/interface_Pred_all_risk)，根據我們所製作的 [使用者辭典](https://github.com/Jason901008/Research-Project/blob/main/jieba%20%E6%96%B7%E8%A9%9E/Ws_Dictionary.txt) 進行斷詞<br>
斷詞：[Code](https://github.com/Jason901008/Research-Project/blob/main/jieba%20%E6%96%B7%E8%A9%9E/jieba_WS.py)<br>

* 隨機森林樹模型<br>
訓練方式：[Random Forest](https://github.com/Jason901008/Research-Project/blob/main/%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97%E6%A8%B9%E6%A8%A1%E5%9E%8B/Random_Forest.pdf)<br>
訓練資料：[Period train](https://github.com/Jason901008/Research-Project/tree/main/%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97%E6%A8%B9%E6%A8%A1%E5%9E%8B/Period_train)<br>
測試資料：[Period test](https://github.com/Jason901008/Research-Project/tree/main/%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97%E6%A8%B9%E6%A8%A1%E5%9E%8B/Period_test)<br>
分類模型：[Code](https://github.com/Jason901008/Research-Project/blob/main/%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97%E6%A8%B9%E6%A8%A1%E5%9E%8B/%E5%88%86%E9%A1%9E%E6%A8%A1%E5%9E%8B/Case_Classifier.py)<br>
回歸模型：[Code](https://github.com/Jason901008/Research-Project/blob/main/%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97%E6%A8%B9%E6%A8%A1%E5%9E%8B/%E5%9B%9E%E6%AD%B8%E6%A8%A1%E5%9E%8B/YearFine_Regression.py)<br>

* TF-IDF與詞嵌入<br>
訓練方式：[Similar Judgments](https://github.com/Jason901008/Research-Project/blob/main/%E7%9B%B8%E4%BC%BC%E5%88%A4%E6%B1%BA%E6%9B%B8/Similar_Judgments.pdf)<br>
訓練資料：[sim data](https://github.com/Jason901008/Research-Project/tree/main/%E7%9B%B8%E4%BC%BC%E5%88%A4%E6%B1%BA%E6%9B%B8/sim_data)<br>
相似判決書：[Code](https://github.com/Jason901008/Research-Project/blob/main/%E7%9B%B8%E4%BC%BC%E5%88%A4%E6%B1%BA%E6%9B%B8/Retrieve_VerdictSimilar_Verdict.py)<br>









