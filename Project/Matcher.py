import pandas as pd
import numpy as np
import spacy
from spacy.matcher import Matcher
import re


def get_chunks(limit, total):
    '''This function splits a big chunk into smaller chunks of which size is equal to limit'''
    current = 0
    chunks = []
    while current < total:
        chunks.append((current, current + limit))
        current += limit
    chunks.append((current, total))
    return chunks


def clean_text(text):
    '''This funtion removes punctuation and clean text'''
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\bnt\b', 'not', text)
    text = re.sub(r'\\s{2,}', r'\.', text)
    return text


def get_features(text):
    '''This function extracts meaningful phrases using spacy rule-based models'''
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)

    p1 = [{'POS': 'ADJ'}, {'POS': 'NOUN', 'OP': '*'}, {'POS': 'NOUN'}]
    p2 = [{'POS': 'NOUN', 'OP': '*'}, {'POS': 'NOUN'}, {'POS': 'ADJ'}]
    matcher.add('features', [p1, p2], greedy='LONGEST')
    matches = matcher(doc)
    features = ''
    for match_id, start, end in matches:
        span = doc[start: end]
        features += (span.text) + '\n'
    return features


def write_chunks(infile, outfile, chunks, chunk_part):
    '''Reads a specific range of lines from input file, extract the features, and append them to the outfile'''
    with open(infile, "r", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if i in range(chunks[chunk_part][0], chunks[chunk_part][1]):
                with open(outfile, "a", encoding="utf-8") as extracted:
                    extracted.write(get_features(l))
            else:
                break


df = pd.read_csv('tripadvisor_hotel_reviews.csv')
df['Review'] = np.vectorize(clean_text)(df['Review'])

# with open('reviews.txt', "w", encoding="utf-8") as f:
#     f.writelines("%s\n" % l for l in df['Review'].tolist())

chunk_numbers = 20
with open('reviews.txt', "r", encoding="utf-8") as f:
    line_numbers = sum(1 for _ in f)
    chunks = get_chunks(line_numbers // chunk_numbers, line_numbers)

# for i in range(len(chunks)):
#     write_chunks('reviews.txt', 'extracted reviews.txt', chunks, i)
write_chunks('reviews.txt', 'extracted reviews.txt', chunks, 22)
