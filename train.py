import pandas as pd
import os
import math
import pickle
from xgboost import XGBClassifier
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
    mask=data['Make'].isin(top2)
    data.loc[mask,col]=1
    data['Body Code']=0
    data['Body Style']=data['Body Style'].str.upper()#Makes sure that everything is standardized
    body_list=(data['Body Style'].value_counts().index)
    body_length=len(body_list)
    body_count=0
    print (body_length)
    col='Body Code'
    print ("Step4")
    while body_count<body_length:#Assigns a value to ever Body Style
        print (body_count)
        body=str(body_list[body_count])
        mask=data['Body Style']==body
        data.loc[mask,col]=body_count+1
        body_count=body_count+1
    print ("Step5")
    data['RP State Plate']=data['RP State Plate'].str.upper()#Makes sure that everything is standardized
    data['State Code']=0
    state_list=(data['RP State Plate'].value_counts().index)
    state_length=len(state_list)
    state_count=0
    col='State Code'
    print (state_length)
    while state_count<state_length:#Assigns a value to ever State
        print (state_count)
        state=str(state_list[state_count])
        mask=data['RP State Plate']==state
        data.loc[mask,col]=state_count+1
        state_count=state_count+1
    print ("Step6")
    data['Color']=data['Color'].str.upper()#Makes sure that everything is standardized
    data['Color Code']=0
    color_list=(data['Color'].value_counts().index)
    color_length=len(color_list)
    color_count=0
    col='Color Code'
    print (color_length)
    while color_count<color_length:#Assigns a value to ever Color
        print (color_count)
        color=str(color_list[color_count])
        mask=data['Color']==color
        data.loc[mask,col]=color_count+1
        color_count=color_count+1
    y=data['Top25']
    x=data[['Color Code','State Code','Body Code']]
    clf=XGBClassifier(max_depth=10,n_estimators=10,gamma=5)#trains the model
    clf=clf.fit(x,y)
    print (str(cwd)+"\overallcarmodel.pkl")
    with open (str(cwd)+"\overallcarmodel.pkl",'wb') as fm:#saves the model for later use
        pickle.dump(clf,fm)
    fm.close()
    print ("Step10")
    return 0
import os.path
cwd=os.getcwd()
cwd=str(cwd)
print (cwd+'\overallcarmodel.pkl')
if os.path.exists(cwd+'\overallcarmodel.pkl'):# checks if training was already done if so only score
    return 0
else:
    train()