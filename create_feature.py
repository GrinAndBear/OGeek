import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import Levenshtein
import difflib

def read_file(path,flag,is_test=False):
    fp = open(path, encoding='utf-8')
    dataset = []
    for line in fp.readlines():
        line = line.strip().split('\t')
        if is_test:
            line.append(-1)
        dataset.append(line)
    data = pd.DataFrame(dataset)
    data.columns = ['prefix', 'query_prediction', 'title', 'tag', 'label']
    data['flag']=flag
    return data
train=read_file('data/oppo_round1_train_20180929.txt',0)
valid=read_file('data/oppo_round1_vali_20180929.txt',1)
test=read_file('data/oppo_round1_test_A_20180929.txt',2,True)
data=pd.concat([train,valid,test])
data.label=data.label.astype('int')
print(data.info())


def encode(data):
    la_e=LabelEncoder()
    encode_column=['tag']
    for column in encode_column:
        data[column]=la_e.fit_transform(data[column])
    return data

'''
计算prefix的长度

'''

def prefix_len(x):
    try:
        return len(x)
    except:
        return len(str(x))



data['prefix_len']=data.prefix.apply(prefix_len)



'''
对query_prediction进行处理
返回长度，以及前十的概率没有就是0
返回相似度 
'''


def extract_prob(pred):
    pred = eval(pred)
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    pred_prob_lst=[]
    for i in range(10):
        if len(pred)<i+2:
            pred_prob_lst.append(0)
        else:
            pred_prob_lst.append(pred[i][1])
    return pred_prob_lst


def extract_similarity(lst):
    pred = eval(lst[1])
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    len_prefix = lst[0]
    similarity=[]
    for i in range(10):
        if len(pred)<i+2:
            similarity.append(0)
        else:
            similarity.append(len_prefix/float(len(pred[i][0])))
    return similarity

def levenshtein_similarity(str1,str2):
    return Levenshtein.ratio(str1,str2)

def get_equal_rate(lst):
    pred = eval(lst[1])
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    equal_rate=[]
    for i in range(10):
        if len(pred)<i+2:
            equal_rate.append(0)
        else:
            equal_rate.append(levenshtein_similarity(lst[0],pred[i][0]))
    return equal_rate


data['pred_prob_lst']=data['query_prediction'].apply(extract_prob)
data['similarity']=data[['prefix_len','query_prediction']].apply(extract_similarity,axis=1)
data['equal_rate']=data[['title','query_prediction']].apply(get_equal_rate,axis=1)

def add_pred_similarity_feat(data):
    for i in range(10):
        data['prediction'+str(i)]=data.pred_prob_lst.apply(lambda x:float(x[i]))
        data['similarity' + str(i)] = data.similarity.apply(lambda x: float(x[i]))
        data['equal_rate' + str(i)] = data.equal_rate.apply(lambda x: float(x[i]))
    return data
data=add_pred_similarity_feat(data)



'''
对 title 进行处理
跟prefix 的相关性
跟prediction的相关性
'''
def prefix_title_sim(lst):
    return lst[0]/float(len(lst[1]))

data['prefix_title_sim']=data[['prefix_len','title']].apply(prefix_title_sim,axis=1)

data=encode(data)
data=data.drop(['prefix','title','query_prediction','pred_prob_lst','similarity','equal_rate'],axis=1)




print(data)
print(data.info())

data.to_csv('./data/data.csv',index=None)




