import pandas as pd
import numpy as np
import spacy
from spacy.matcher import Matcher
import re


def get_chunks(limit, total):
    '''Splits a big chunk into equally smaller ones'''
    total = 20491
    limit = 1000
    current = 0
    chunks = []
    while current < total:
        chunks.append((current, current + limit))
        current += limit
    chunks.append((current, total))
    return chunks


def get_features(text):
    '''Uses Spacy rule-based matcher to extract phrases from a text'''
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
            if i in chunks[chunk_part]:
                with open(outfile, "a", encoding="utf-8") as extracted:
                    extracted.write(get_features(l))
            else:
                break


df = pd.read_csv('tripadvisor_hotel_reviews.csv')
df['Review'] = np.vectorize(clean_text)(df['Review'])

# with open('reviews.txt', "w", encoding="utf-8") as f:
#     f.writelines("%s\n" % l for l in df['Review'].tolist())

with open('reviews.txt', "r", encoding="utf-8") as f:
    lines = sum(1 for _ in f)
    chunks = get_chunks(lines // 20, lines)

write_chunks('reviews.txt', 'extracted reviews.txt', chunks, 0)
