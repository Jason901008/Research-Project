# 深度學習在法院判決書上的重要訊息擷取與預測之研究
### 交易內容：[UI介面的交易紀錄](https://mumbai.polygonscan.com/address/0xcc4769A4F0367d884177b041A7cd4E3bEF5Afa21)

* 資料來源<br>
刑事判決書來源：[司法院資料開放平台](https://opendata.judicial.gov.tw/dataset/detail?datasetId=27959)<br>
開放平台使用說明：[Source](https://github.com/Jason901008/Research-Project/blob/main/%E8%B3%87%E6%96%99%E4%BE%86%E6%BA%90/Source.pdf)<br>

* 標籤工具<br>
[label studio下載與使用說明](https://blog.csdn.net/qq_44193969/article/details/123298406)<br>
轉成BIOES格式：[Span to BIOES](https://github.com/Jason901008/Research-Project/blob/main/%E6%A8%99%E7%B1%A4%E5%B7%A5%E5%85%B7/Span_to_BIOES.py)<br>

* 風險詞<br>
需要先下載 [transformers](https://github.com/huggingface/transformers/tree/main/examples)<br>
微調NER模型：[Fine-tune NER Model](https://github.com/Jason901008/Research-Project/blob/main/%E9%A2%A8%E9%9A%AA%E8%A9%9E/Fine-tune_NER_Model.pdf)<br>
雲端 [NER MODEL](https://drive.google.com/drive/folders/1Th6UCs6kKGzA38C7cvFBtrE40LQUy3Qe?usp=drive_link)<br>
判決書原文資料：[Original Data](https://github.com/Jason901008/Research-Project/tree/main/%E9%A2%A8%E9%9A%AA%E8%A9%9E/Result_Record/Original_Data)<br>
NER模型使用：[NER TEST](https://github.com/Jason901008/Research-Project/blob/main/%E9%A2%A8%E9%9A%AA%E8%A9%9E/NER_TEST.py)<br>

* Jieba斷詞<br>
使用上述 NER 模型的[輸出結果](https://github.com/Jason901008/Research-Project/tree/main/jieba%20%E6%96%B7%E8%A9%9E/interface_Pred_all_risk)，根據我們所製作的[使用者辭典]()進行斷詞，















