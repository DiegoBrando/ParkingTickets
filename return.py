import math
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import confusion_matrix
import os
import sys
def add_missing_columns( data, columns ):
    missing_cols = set( columns ) - set( data.columns )
    for z in missing_cols:
        data[z] = 0
	return data
def dummy_columns( data, columns ):  

    data=add_missing_columns( data, columns )

    assert( set( columns ) - set( data.columns ) == set())

    extra_cols = set( data.columns ) - set( columns )
    if extra_cols:
        print ("added columns: ", extra_cols)
    data = data[ columns ]
    return data

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
    data=pd.read_csv(str(cwd)+"\traineddatacar.csv")#reads in non corrupted data to make sure categorized values are the same
    print ("StepC")
    data= data.loc[:, ~data.columns.str.startswith('Unn')]]
    
    datatest=datatest[['Color','Body style']]
    datatest=pd.get_dummies(datatest, prefix=['Body Style', 'Color'], columns=['Body Style', 'Color'])

    x=datatest
    print ("StepG")
    x= x.loc[:, ~x.columns.str.startswith('Unn')]]
    data=data.loc[:, ~data.columns.str.startswith('Unn')]]
    x=dummy_columns(x,data.columns)
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
if (os.path.exists(str(cwd)+'\overallcarmodel.pkl') and os.path.exists(str(cwd)+"\traineddatacar.csv")):# checks if training was already done if so only score
    datareturn=score()
else:# if training was not already done trains model then scores
    train()
    datareturn=score()
print ("Processing Complete, Prediction at "+cwd+"\output.json")
    
