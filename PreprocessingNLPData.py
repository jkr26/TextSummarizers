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
    def __init__(self):
        """
        """
        self.raw_data = read_wiki2010_corpus_first_million()
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

        
def read_wiki2010_corpus_first_million():
    cnxn = psycopg2.connect("host='localhost' dbname='TextSummary' user='PythonExecutor' password='2Cons!stent'")
    cursor = cnxn.cursor()
    try:    
        cursor.execute("""
                       SELECT TitleAndSummary, LongDescription
                       FROM SummaryDescriptionPairs
                       WHERE CorpusName = 'Wikipedia2010Corpus'
                       """
                       )
        summary_description_pairs = list(set(cursor.fetchmany(1e2)))
        summary_description_pairs = [(pair[0], pair[1]) for pair in summary_description_pairs
                                     if (pair[0] and pair[1] and len(pair[1].split())<5000)]
    
    finally:
        cursor.close()
        cnxn.close()
    return summary_description_pairs

#if __name__=="__main__":
#    dset = WikipediaCorpusFirstMillion()
#    loader = DataLoader(dset, num_workers=8)