import numpy as np
import xml.etree.ElementTree as ET
import json
import pandas as pd
from collections import Counter, defaultdict

import os

def to_df(word_counts):
    wordcount_df = pd.DataFrame.from_dict(word_counts, orient = 'index').reset_index(
                      ).rename(columns = {'index': 'Word', 0: 'Count'}).sort_values(by = "Count", ascending=False)
    return wordcount_df

def consolidate_dict2(corpus):
    counts = Counter()

    for t in corpus:
        counts.update(t)

    term_freq = list(map(lambda x: {t: x.count(t) for t in x}, corpus))
    return counts, term_freq

if __name__ == "__main__":

    # Replace with Absolute Path?
    train_path = 'dataset/ABSA-15_Laptops_Train_Data.xml'
    
    # Two different paths for different nodes: sentences and opinions
    node = 'Review/sentences/sentence'
    opinion_node = 'Review/sentences/sentence/Opinions/Opinion'

    tree = ET.parse(train_path)
    root = tree.getroot()
    train_sentences = root.findall(node)
    root = tree.getroot()
    train_opinions = root.findall(opinion_node)

    train_text = [sent[0].text for sent in train_sentences]
    opinions = set([op.attrib['category'] for op in train_opinions])
    with open('laptop_cats.txt', 'w') as f:
        f.write(str(opinions))

    with open('laptop_map.json', 'r') as file:
        category_mapping = json.load(file)

    mapped_opinions = [category_mapping[op] for op in opinions]
    unique_categories = {key:[] for key in set(mapped_opinions)}

    for sent in train_sentences:
        if len(sent) > 1 and len(sent[1]) > 0:
            if sent[1].tag == 'Opinions':
                aspect_cats = sent[1]
            else:
                aspect_cats = sent[2] 
                
            for opinions in aspect_cats:
                dict = opinions.attrib
                if dict['category'] in category_mapping:
                    unique_categories[category_mapping[dict['category']]].append(sent[0].text.lower().split())

    topic_dicts = {}
    topic_dfs = {}
    tf_cat = {}

    # for generating seed vectors for categories
    for cat in unique_categories:
        topic_dicts[cat], tf_cat[cat] = consolidate_dict2(unique_categories[cat])
        topic_dfs[cat] = to_df(topic_dicts[cat])
        topic_dfs[cat].to_csv(cat + '_wordcounts.csv', index = False)

