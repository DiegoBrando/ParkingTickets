import math
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import confusion_matrix
import os
import sys

#Print steps are there to make monitor performance
def train():
    print ("Step1")
    data=pd.read_csv('https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv')#bringing in training data
    print ("Step2")
    data['Make']=data['Make'].fillna("N/A")
    data=data[data['Make']!="N/A"]# removes all corrupted data
    cwd=os.getcwd()
    data.to_csv(str(cwd)+"\parking_citation.csv")# saves uncorrupted data for use later
    print ("Step3")
    data['Make']=data['Make'].str.upper()
    top= (data['Make'].value_counts().index)
    top2=top[:25]
    data['Top25']=0
    col='Top25'
	#mask=~(data['Make'].isin(top2))
    mask=data['Make'].isin(top2)
    data.loc[mask,col]=1
    data=data[['Color','Body Style','Top25']]
	#data['TOP25']=data['TOP25'].replace({0:1, 1:0})
    data['Body Style']=data['Body Style'].str.upper()#Makes sure that everything is standardized
    body_list=(data['Body Style'].value_counts().index)
    body_list=body_list[:int(len(body_list)*.70)]#save on resources only use top 70% of body styles
    print ("Step4")

    data['Color']=data['Color'].str.upper()#Makes sure that everything is standardized
    color_list=(data['Color'].value_counts().index)
    color_list=color_list[:int(len(color_list)*.70)]
    print ("Step5")
    data=data[((data['Color'].isin(color_list)) & (data['Body Style'].isin(body_list)))].reset_index()# train only on bodys and colors that are in the top 70%
    y=data['Top25']
    x=data.drop(columns=['Color','Body Style','Top25'])
    x= x.loc[:, ~x.columns.str.startswith('Unn')]]
    clf=XGBClassifier(max_depth=10,n_estimators=10,gamma=5)#trains the model
    clf=clf.fit(x,y)
    print (str(cwd)+"\overallcarmodel.pkl")
    with open (str(cwd)+"\overallcarmodel.pkl",'wb') as fm:#saves the model for later use
        pickle.dump(clf,fm)
    fm.close()
    x.to_csv(str(cwd)+"\traineddatacar.csv")
    print ("Step10")
    return 0
import os.path
cwd=os.getcwd()
cwd=str(cwd)
print (cwd+'\overallcarmodel.pkl')
if (os.path.exists(str(cwd)+'\overallcarmodel.pkl') and os.path.exists(str(cwd)+"\traineddatacar.csv")):# checks if training was already done if so only score
    return 0
else:
    train()