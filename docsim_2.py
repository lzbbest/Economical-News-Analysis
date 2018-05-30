# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re,time,glob
import jieba_fast as jieba
from gensim import corpora, models, similarities
import logging
import gensim
from gensim.similarities import Similarity
import thulac
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#thu1 = thulac.thulac(user_dict = 'E:\\codetest\\fintech\\topic2\\THUOCL_caijing.txt',seg_only=True, filt=True)
dicts = glob.glob('E:\\codetest\\fintech\\topic2\\userdict\\*')
for d in dicts:
    print(d)
    jieba.load_userdict(d)

def readFile():
    train_data = 'E:\\codetest\\fintech\\topic2\\train_data.csv'
    test_data = 'E:\\codetest\\fintech\\topic2\\test_data.csv'
    stop = 'E:\\codetest\\fintech\\topic2\\stopwords.dat'
    trainD = pd.read_csv(train_data, skiprows=0,index_col='id')
    testD = pd.read_csv(test_data,index_col='id')
    stopword = [line.strip() for line in open(stop,encoding='utf-8').readlines()]
    return trainD,testD,stopword

def tcutword(data,stopword):
    corpora_documents = []
    for i in data.index:
        text = data.loc[i].values[0].strip()
        text = re.sub('[\"*\【\】\[\]\s*]','',text) # sub Special symbol
        text =re.sub('\([a-zA-z]+://[^\s]*\)','',text) # substitute URL
        text = re.sub('\d+\.*\d*','',text)
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", '',text)        
        #cuting = jieba.cut(text)
        #cuting = ' '.join(cuting)
        temp = list(jieba.cut(text,HMM=True))
        
        #temp=thu1.cut(text,text=True).split()
        word_list = temp
        '''
        word_list = []  
        for word in temp:
            if word not in stopword:  
                word_list.append(word)  
        #text = ' '.join(temp)
        '''
        corpora_documents.append(word_list)
    dictionary = corpora.Dictionary(corpora_documents)
    corpus = [dictionary.doc2bow(ttext) for ttext in corpora_documents]
    similarity = similarities.Similarity('-Similarity-index', corpus, num_features=99999999)
    return dictionary,similarity

def testcut(testD,stopword,dictionary,similarity):
    s_id,t_id = [],[]
    for i in testD.index:
        text = testD.loc[i].values[0].strip()
        text = re.sub('[\"*\【\】\[\]\s*]','',text) # sub Special symbol
        text =re.sub('\([a-zA-z]+://[^\s]*\)','',text) # substitute URL
        text = re.sub('\d+\.*\d*','',text)
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", '',text)        
        #cuting = jieba.cut(text)
        #cuting = ' '.join(cuting)
        temp = list(jieba.cut(text,HMM=True))
        #temp=thu1.cut(text,text=True).split()
        word_list = temp
        '''###
        word_list = []
        for word in temp:
            if word not in stopword:  
                word_list.append(word)  
        '''###
        test_corpus = dictionary.doc2bow(word_list)
        similarity.num_best = 21
        temp_id = []
        [temp_id.append(int(item[0])+1) for item in similarity[test_corpus]]
        if i not in temp_id:
            t_id.extend(temp_id[:20])
        else:
            temp_id.remove(i)
            t_id.extend(temp_id)
        [s_id.append(i) for j in range(20)]
    dfre = pd.DataFrame({'source_id':s_id,'target_id':t_id})
    return dfre


if __name__ == '__main__':   
    trainD,testD,stopword = readFile()
    dictionary,similarity = tcutword(trainD,stopword)
    df = testcut(testD,stopword,dictionary,similarity)
    currentTime = time.strftime("%Y%m%d.%H%M%S", time.localtime())
    resultPath = currentTime + "-Result"
    df.to_csv(resultPath,sep='\t',index=False)







