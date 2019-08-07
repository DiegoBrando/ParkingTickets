import math
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import confusion_matrix
data=pd.read_csv("/home/aoakey/car/parking_citationstop25.csv", encoding = "ISO-8859-1")
print (data.columns.values)
#http://fastml.com/how-to-use-pd-dot-get-dummies-with-the-test-set/
bodyl=data['Body Style'].value_counts().index
bodyl=bodyl[:int(len(bodyl)*.70)]
colorl=data['Color'].value_counts().index
colorl=colorl[:int(len(colorl)*.70)]

data=data[((data['Color'].isin(colorl)) & (data['Body Style'].isin(bodyl)))].reset_index()
data['TOP25']=data['TOP25'].replace({0:1, 1:0})
y=data['TOP25']
data=data[['Body Style', 'Color']]
data=pd.get_dummies(data, prefix=['Body Style', 'Color'], columns=['Body Style', 'Color'])
data.to_csv("/home/aoakey/car/traindataset.csv")
print ('Yay')
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=.2,random_state=42)
clf=XGBClassifier(max_depth=20,n_estimators=10,gamma=5)
clf=clf.fit(X_train,y_train)

with open ("/home/aoakey/car/overallcarmodelnocomboreversemd20ne10g5.pkl",'wb') as fw:
   pickle.dump(clf,fw)
fw.close()
pred=clf.predict_proba(X_test)
f=open("/home/aoakey/car/caronlycomboreverseXGmin"+str(20)+"NE"+str(10)+"G"+str(5)+".tsv",'w')

f.write("ALGORITHM\tThreshold\tTruePositive\tTrueNegative\tFalsePositive\tFalseNegative\n")
cutoff=0.0
X_test['Predicted']=pred[:,1]
X_test['Top25']=0
col='Top25'
while cutoff<1:
    mask=X_test['Predicted']>=cutoff
    X_test.loc[mask,col]=1
    confusionmatrix = confusion_matrix(y_test, X_test['Top25'])


    tn = confusionmatrix [0][0]
    fn = confusionmatrix [1][0]
    tp = confusionmatrix [1][1]
    fp = confusionmatrix [0][1]

    cutoff=cutoff+.01
    f.write("XGBMD"+str(20)+"NE"+str(10)+"G"+str(5)+"\t"+str(cutoff)+"\t"+str(tp)+"\t"+str(tn)+"\t"+str(fp)+"\t"+str(fn)+"\n")
f.close()


