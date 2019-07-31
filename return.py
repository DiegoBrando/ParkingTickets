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

def score():
    print ("StepA")
    cwd=os.getcwd()
    cwd=str(cwd)
    try:# checks if input file exists
        datatest=pd.read_json(cwd+"\input.json")
        
    except:
	    print ('File '+cwd+"\input.json can not be found.")
        sys.exit()
    with open (str(cwd)+"\overallcarmodel.pkl",'rb') as fm:#loads trained model
        clf=pickle.load(fm)
    fm.close()
    print ("StepB")
    data=pd.read_csv(str(cwd)+"\parking_citation.csv")#reads in non corrupted data to make sure categorized values are the same
    print ("StepC")
    data['Body Style']=data['Body Style'].str.upper()
    datatest['Body Style']=datatest['Body Style'].str.upper()
    body_list=(data['Body Style'].value_counts().index)
    body_length=len(body_list)
    body_count=0
    datatest['Body Code']=0
    col='Body Code'
    while body_count<body_length:#assigns values same values for body type as training
        body=str(body_list[body_count])
        mask=datatest['Body Style']==body
        datatest.loc[mask,col]=body_count+1
        body_count=body_count+1
    print ("StepD")
    data['RP State Plate']=data['RP State Plate'].str.upper()
    datatest['RP State Plate']=datatest['RP State Plate'].str.upper()
    datatest['State Code']=0
    state_list=(data['RP State Plate'].value_counts().index)
    state_length=len(state_list)
    state_count=0
    col='State Code'
    print (state_length)
    while state_count<state_length:#assigns values same values for state as training
        state=str(state_list[state_count])
        mask=datatest['RP State Plate']==state
        datatest.loc[mask,col]=state_count+1
        state_count=state_count+1
    print ("StepE")
    datatest['Color']=datatest['Color'].str.upper()
    data['Color']=data['Color'].str.upper()
    datatest['Color Code']=0
    color_list=(data['Color'].value_counts().index)
    color_length=len(color_list)
    color_count=0
    col='Color Code'
    while color_count<color_length:#assigns values same values for color as training
        color=str(color_list[color_count])
        mask=datatest['Color']==color
        datatest.loc[mask,col]=color_count+1
        color_count=color_count+1
    print ("StepF")
    
    x=datatest[['Color Code','State Code','Body Code']]
    print ("StepG")
    pred=clf.predict_proba(x)#predicts probability for it being car being in top25
    y=pred[:,1]
    datatest['Top25']=y
    print ("StepH")
    datatest.to_json(cwd+"\output.json")# outputs file
    return datatest
import os.path
cwd=os.getcwd()
cwd=str(cwd)
print (cwd+'\overallcarmodel.pkl')
if os.path.exists(cwd+'\overallcarmodel.pkl'):# checks if training was already done if so only score
    datareturn=score()
else:# if training was not already done trains model then scores
    train()
    datareturn=score()
print ("Processing Complete, Prediction at "+cwd+"\output.json")
    
