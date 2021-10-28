import pandas as pd
import numpy as np
import spacy
from spacy.matcher import Matcher
import re


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\bnt\b', 'not', text)
    text = re.sub(r'\\s{2,}', r'\.', text)
    return text


def get_features(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)

    p1 = [{'POS': 'ADJ'}, {'POS': 'NOUN', 'OP': '*'}, {'POS': 'NOUN'}]
    p2 = [{'POS': 'NOUN', 'OP': '*'}, {'POS': 'NOUN'}, {'POS': 'ADJ'}]
    matcher.add('features', [p1, p2], greedy='LONGEST')
    matches = matcher(doc)
    features = []
    for match_id, start, end in matches:
        span = doc[start: end]
        features.append(span.text)
    return features


df = pd.read_csv('tripadvisor_hotel_reviews.csv')
df['Review'] = np.vectorize(clean_text)(df['Review'])

with open('reviews.txt', "w", encoding="utf-8") as f:
    f.writelines("%s\n" % l for l in df['Review'].tolist())
