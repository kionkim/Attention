# RNN
Material for MODU

# Requirements

* Make sure that you have python==3.6.5

Requirements are 

---

mxnet==1.3.0

tqdm==4.24.0

spacy

pandas==0.23.4

scikit-learn==0.20.0

scipy==1.1.0

sklearn==0.0

gluonnlp==0.4.1

---


To install requirements, issue the following code.

```
pip install -r requirements.txt
```

# Usage

Before run models please issue the following code.

```
python -m spacy download en
```

There are 6 files in ./python. File tree looks like

├── 1_sentiment_RNN_SA.py

├── 2_sentiment_RN.py

├── 3_sentiment_RN_SA.py

├── 4_date_formatter_att.py

├── 5_date_formatter_mul_att.py

├── 6_date_formatter_add_att.py

└── utils.py


# Etc.

For more details on the models, please read the following blogs

* Sentiment analysis using Self-attention
https://medium.com/apache-mxnet/sentiment-analysis-via-self-attention-with-mxnet-gluon-dc774d38ba69

* Self-attention: A clever compromise
https://medium.com/@kion.kim/self-attention-a-clever-compromise-4d61c28b8235
