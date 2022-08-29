from cgitb import text
import csv
import pandas as pd
import nltk
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
import urllib.request

def preprocess(example):
    new_example = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_example.append(t)
    return " ".join(new_example)

df = pd.read_csv("Google_Climate_SouthKorea_tweets_Korean_New.csv")
example = df['Tweet'][16]


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#Running example on ROBERTA
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'Roberta_Neg' : scores[0],
    'Roberta_Neu' : scores[1],
    'Roberta_Pos' : scores[2]
}

def offensive_score_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

print(scores_dict)  


#Whole Data set on ROBERTA
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Translation']
    myid = row['User']
    #roberta_result = hatespeech_scores_roberta(text)
    res[myid] = offensive_score_roberta(text)

print(pd.DataFrame(res).T)
pd.DataFrame(res).T.to_csv('Climate_NEW_Result_sentiment_korean.csv')