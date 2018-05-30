# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re,os
import jieba_fast as jieba
from gensim import corpora, models, similarities
import logging
from gensim.models import word2vec
import gensim
from gensim.models.doc2vec import Doc2Vec, LabeledSentence  
from gensim.similarities import Similarity

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def readFile():
    train_data = 'E:\\codetest\\fintech\\topic2\\train_data.csv'
    test_data = 'E:\\codetest\\fintech\\topic2\\test_data.csv'
    stop = 'E:\\codetest\\fintech\\topic2\\stopword.txt'
    trainD = pd.read_csv(train_data, skiprows=0 ,index_col='id')
    testD = pd.read_csv(test_data,index_col='id')
    stopword = [line.strip() for line in open(stop).readlines() ]
    return trainD,testD,stopword

def cutword(data,stopword):
    for i in data.index:
        text = data.loc[i].values[0].strip()
        text = re.sub('[\"*\【\】\[\]\s*]','',text) # sub Special symbol
        text =re.sub('\([a-zA-z]+://[^\s]*\)','',text) # substitute URL
        text = re.sub('\d+\.*\d*','',text)
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", '',text)        
        #cuting = jieba.cut(text)
        #cuting = ' '.join(cuting)
        temp = list(jieba.cut(text,HMM=True))
        outStr = ''
        for word in temp:
            if word not in stopword:  
                outStr += word  
                outStr += ' '
        #text = ' '.join(temp)
        text = outStr
        data.loc[i].values[0] = text
    return data

def docvec(trainD):
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    data = []
    for i in trainD.index:
        text = trainD.loc[i].values[0]
        word_list = text.split()
        document = TaggededDocument(word_list, tags=[i])
        data.append(document)  
    
    model = Doc2Vec(data,min_count=1, window = 6, size = 100, sample=1e-3, negative=5, workers=4)
    model.train(data, total_examples=model.corpus_count, epochs=10) 
    model.save('model1')
    return model

def test(model,testD):
    s_id,result = [],[]
    for i in testD.index:
        text = testD.loc[i].values[0]
        infer = model.infer_vector(text.split())  
        sims = model.docvecs.most_similar([infer], topn=20)
        [result.append(t_id[0]) for t_id in sims]
        [s_id.append(i) for j in range(20)]
    dfresult = pd.DataFrame({'source_id':s_id,'target_id':result}) 
    return dfresult


if __name__ == '__main__':   
    trainD,testD,stopword = readFile()
    #trainD = cutword(trainD,stopword)
    testD = cutword(testD,stopword)
    #model = docvec(trainD)
    #df = test(model,testD)
    #df.to_csv('result.txt',sep='\t',index=False)
    
'''

    pattern = re.compile('DP4=\d+,\d+,\d+,\d+;',re.I)
    t = pattern.search(s).group()[4:-1].split(',')
    tList = []
    for i in friends:# 获取个性签名,过滤emoji
        signature = i["Signature"].strip('\"\"\"').replace("【", ' ').replace("】", ' ').replace("emoji", "")
        rep = re.compile("1f\d.+")
        signature = rep.sub("", signature)
        tList.append(signature)

    text = "".join(tList)#拼接字符串
    wordlist=jieba.cut(text)#分词
    wl=" ".join(wordlist)
'''




