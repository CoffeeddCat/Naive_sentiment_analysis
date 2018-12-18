# -*- coding: utf-8 -*-
"""
Here is Word Embedding Procedure:
1.Skip Space (including ' ' and '\n')
2.Word segmentation
3.Word Unabbrivation (I'll ->I will)
4.Embedding (Getting a n*K matrix , n : # of words , K : Embedding feature size)
"""

import gensim
import os
import numpy as np
from nltk import sent_tokenize, word_tokenize
import jieba
from config import *

def load_word2vec_model(language):
    if language==CN:
        #model based on wiki training set : feature size=500
        #return gensim.models.Word2Vec.load(os.path.join(Word_Embedding_Dir,'wiki_word2vec','word2vec_wiki.model')) # Feature_Size = 500    
        
        #model based on wechat training set : feature size=256 
        return gensim.models.Word2Vec.load(os.path.join(Word_Embedding_Dir, 'word2vec_wx')) # Feature_Size = 256 
    elif language==EN:
        return gensim.models.Word2Vec.load(os.path.join(Word_Embedding_Dir, 'word2vec_en_trained.txt')) # Feature_Size = 100

# skip ' ' and '\n' in chinese texts
def skip_space(sentence):
    while len(sentence) > 0 and (sentence[0] == '\n' or sentence[0] == ' '):
        sentence = sentence[1:]
    while len(sentence) > 0 and (sentence[-1] == '\n' or sentence[-1] == ' '):
        sentence = sentence[:-1]
    return sentence

#abbr. -> unabbr.   (e.g. I'll = I will)
def word_unabbrivated(words):
    for i in range(len(words)):
        if words[i]=="'ll":
            words[i]="will"
        if words[i] == "'ve":
            words[i] = "have"
        elif words[i] == "'m":
            words[i] = "am"
        elif words[i] == "'d":
            if i+1<len(words):
                if words[i+1]=="rather" or words[i+1]=="like" or words[i+1]=="be":
                    words[i]="would"
                else: 
                    words[i]="had"
            else:
                words[i]="would"
        elif words[i] == "n't":         
            if i > 0 and words[i-1] == "wo":
                words[i-1] = "will"
            words[i] = "not"
    return words

def word_segmentation(sentence,language):
    if language==CN:
        sentence=skip_space(sentence)
        return jieba.lcut(sentence)
    elif language==EN:
        return word_unabbrivated(word_tokenize(sentence))

def embedding(model,txt,language):
    words=word_segmentation(txt,language)
    print(words)
    embedding_mat = np.zeros((len(words), Embedding_dim))
    i=0
    for word in words:
        try:
            embedding_mat[i]=model[word]
        except KeyError:
            embedding_mat[i]=0.0
        i=i+1
    return embedding_mat


# Testing 
if __name__ == '__main__':
    model_en = load_word2vec_model(EN)
    # model_cn = load_word2vec_model(CN)
    text="it's very good"
    mat=embedding(model_en,text,CN)
    print(mat.shape)
    print (mat)
    #print (model_en['good'],model_en.most_similar('good',topn=5)) 
    #print (model_cn.most_similar('?',topn=10))
    