#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import psycopg2
import string
"""
Created on Fri Dec 29 18:38:04 2017

@author: jkr
"""



class WikipediaCorpusFirstMillion(Dataset):
    def __init__(self, num):
        """
        """
        self.raw_data = read_wiki2010_corpus_first_million(num)
        self.raw_data = _remove_punctuation(self.raw_data)

    def __getindex__(self,idx):
        return self.raw_data[idx]
    
    def __len___(self):
        return len(self.raw_data)
    
def _remove_punctuation(data_list):
    table = str.maketrans("","", string.punctuation)
    to_return = []
    for datum in data_list:
        summary = datum[0]
        long_description = datum[1]
        summary_list = []
        description_list = []
        for word in summary.split():
            summary_list.append(word.translate(table))
        for word in long_description.split():
            description_list.append(word.translate(table))
        to_return.append((summary_list, description_list))
    return to_return

        
def read_wiki2010_corpus_first_million(num):
    cnxn = psycopg2.connect("host='localhost' dbname='TextSummary' user='PythonExecutor' password='2Cons!stent'")
    cursor = cnxn.cursor()
    try:    
        cursor.execute("""
                       SELECT TitleAndSummary, LongDescription
                       FROM SummaryDescriptionPairs
                       WHERE CorpusName = 'Wikipedia2010Corpus'
                       """
                       )
        summary_description_pairs = list(set(cursor.fetchmany(num)))
        summary_description_pairs = [(pair[0], pair[1]) for pair in summary_description_pairs
                                     if (pair[0] and pair[1] and len(pair[1].split())<3000)]
    
    finally:
        cursor.close()
        cnxn.close()
    return summary_description_pairs

class DataStatistics:
    def __init__(self, name, max_target_vocab=20000):
        self.name = name
        self.targetword2index = {"SOS":0, "EOS":1,'<unk>':2}
        self.targetword2count = {'<unk>':10000, 'SOS':10000, 'EOS':10000}
        self.targetindex2word = {0: "SOS", 1: "EOS", 2:'<unk>'}
        self.n_words_target = 3  # Count SOS and EOS and unk
        self.max_length = 0
        self.glove_dict = create_glove_dict()
        self.glove_vector_size = len(self.glove_dict['the'])
        self.max_target_vocab = max_target_vocab
        self.extended_vocab = None

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)
                
    def updateMaxLength(self, sentence):
        for word in sentence:
            if len(sentence)+3>self.max_length:
                self.max_length = len(sentence)+3
                print(str(len(sentence)+3))

    def addWord(self, word):
        word = word.lower()
        if word not in self.targetword2index:
            self.targetword2index[word] = self.n_words_target-2
            self.targetword2count[word] = 1
            self.targetindex2word[self.n_words_target] = word.lower()
            self.n_words_target += 1
        else:
            self.targetword2count[word] += 1
    
    def restrictVocab(self):
        i2w = pd.DataFrame.from_dict(self.targetindex2word,orient='index').reset_index(drop=False)
        i2w.columns=['Idx', 'Word']
        w2c = pd.DataFrame.from_dict(self.targetword2count,orient='index').reset_index(drop=False)
        w2c.columns=['Word', 'Count']
        df = i2w.merge(w2c, how = 'inner', on='Word')
        df.sort_values(by='Count', ascending=False, inplace=True)
        restricted = df.head(self.max_target_vocab)
        restricted.reset_index(inplace=True, drop=False)
        i2w = {idx:row['Word'] for idx, row in restricted.iterrows()}
        w2c = {row['Word']:row['Count'] for idx, row in restricted.iterrows()}
        w2i = {row['Word']:idx for idx, row in restricted.iterrows()}
        self.targetindex2word = i2w
        self.targetword2index = w2i
        self.targetword2count = w2c
        self.n_words_target = len(restricted)
    

def create_glove_dict():
    glove_dict={}
    with open('/home/jkr/GloVe-1.2/vectors.txt') as file:
        for line in file:
            ls = line.split()
            word=ls[0]
            ls.pop(0)
            vec = ls.copy()
            idx = 0
            for w in vec:
                vec[idx] = float(w)
                idx+=1
            glove_dict[word] = vec
    return glove_dict


def generateWikiCorpusTrainingPairsAndDataStatistics(num):
    Data = WikipediaCorpusFirstMillion(num)
    training_pairs = Data.raw_data
    del Data
    training_pairs =[([w.lower() for w in pair[0]], [w.lower() for w in pair[1]]) for pair in training_pairs]

    ds = DataStatistics('WikipediaCorpus', max_target_vocab=20000)
    for pair in training_pairs:
        ##0th element is the summary--1st is the long description.
        ##A little backwards perhaps
        ds.addSentence(pair[0])
        ds.updateMaxLength(pair[1])
    ds.restrictVocab()

    return training_pairs, ds


#if __name__=="__main__":
#    dset = WikipediaCorpusFirstMillion()
#    loader = DataLoader(dset, num_workers=8)