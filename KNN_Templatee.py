#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

titanic_data=pd.read_csv("titanic.csv")

#print(titanic_data)

titanic_data.drop(["PassengerId","Name","Ticket","Cabin","Embarked"],axis=1,inplace=True)

#print(titanic_data)

titanic_data["Sex_M"] = titanic_data['Sex'].map({"male" :1, "female" : 0})
titanic_data.drop("Sex", axis = 1, inplace = True)

#print(titanic_data)

#import seaborn as sns
#sns.heatmap(titanic_data.corr(),cmap='YlGnBu')
#plt.show()

titanic_data["Age"].fillna(value=titanic_data["Age"].mean(),inplace=True)

#print(titanic_data)

X = titanic_data.drop(["Survived"],axis=1)
y = titanic_data["Survived"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.25,random_state=0)

clf= KNeighborsClassifier(n_neighbors=17)
model=clf.fit(X_train,y_train)
y_predicted=clf.predict(X_test)

#print("Accuracy: %2f" % metrics.accuracy_score(y_test,y_predicted))
#print("Recall: %2f" % metrics.recall_score(y_test,y_predicted, average="macro"))
#print("Precision: %2f" % metrics.precision_score(y_test, y_predicted, average="macro"))
#print("F1: %2f" % metrics.f1_score(y_test, y_predicted, average="macro"))

A=[]
for k in range(1,201):
    for p in range(1,4):
        for w in ["uniform","distance"]:
            clf=KNeighborsClassifier(n_neighbors=k,weights=w,p=p)
            model=clf.fit(X_train,y_train)
            y_predicted=clf.predict(X_test)
            f1=metrics.f1_score(y_test, y_predicted, average="macro")
            acc=metrics.accuracy_score(y_test,y_predicted)
            rec=metrics.recall_score(y_test,y_predicted, average="macro")
            pr=metrics.precision_score(y_test, y_predicted, average="macro")
            A.append([k,w,p,f1,acc,rec,pr])

#print(A)

A=pd.DataFrame(A,columns=["k","weights","p","f1","accuracy","recall","precision"])

#print(A)

print(A[A["weights"]=="distance"][A["p"]==2]["f1"].idxmax())

#print(A.iloc[117])

pltdf=A[A["weights"]=="uniform"][A["p"]==1][["k","f1"]]

#print(pltdf)

import plotly.express as px

fig = px.line(pltdf, x="k", y="f1", title='p=1 and weights=uniform (without Age)')
fig.show()