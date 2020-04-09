#(1)GET LIBRARIES
#Import libraries
import pandas as pd
import numpy as np
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


#Load data files
train=pd.read_csv("C:/Users/muska/Desktop/Internship/LoanApprovalProject_101/Data/train_u6lujuX_CVtuZ9i.csv")

#List of column names
list(train)

#Sample of data
train.head(10)

#Types of data columns
train.dtypes

#Summary statistics
train.describe()

#(2)DATA CLEANING AND PREPROCESSING
#Find missing values
train.isnull().sum()

#Impute missing values with mean (numerical variables)
train.fillna(train.mean(),inplace=True) 
train.isnull().sum() 

#Impute missing values with mode (categorical variables)
train.Gender.fillna(train.Gender.mode()[0],inplace=True)
train.Married.fillna(train.Married.mode()[0],inplace=True)
train.Dependents.fillna(train.Dependents.mode()[0],inplace=True) 
train.Self_Employed.fillna(train.Self_Employed.mode()[0],inplace=True)  
train.isnull().sum() 


#Treatment of outliers
train.Loan_Amount_Term=np.log(train.Loan_Amount_Term)

#(3)PREDICTIVE MODELLING

#Create target variable
X=train.drop('Loan_Status',1)
y=train.Loan_Status


#Split train data for cross validation
from sklearn.model_selection import train_test_split
#Percentage of Training data used for modeling (train data will be applied agianst this resulting model)
#The accuracy/prediction of each run will be different because of the above
x_train,x_cv,y_train,y_cv = train_test_split(X,y,test_size=0.2)

#Build dummy variables for categorical variables
x_train=x_train.drop('Loan_ID',axis=1)
x_train=pd.get_dummies(x_train)

tmp = x_cv['Loan_ID']
x_cv=x_cv.drop('Loan_ID',axis=1)
x_cv=pd.get_dummies(x_cv)

#(a)LOGISTIC REGRESSION ALGORITHM
#Fit model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

#Predict values for cv data
pred_cv=model.predict(x_cv)
#Write test results in csv file
predictions=pd.DataFrame({'Loan_ID':tmp,'predictions':pred_cv}).reset_index(drop=True).to_csv('LogisticRegression_Predictions.csv')

#Evaluate accuracy of model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy_score(y_cv,pred_cv) #78.86%
matrix=confusion_matrix(y_cv,pred_cv)
print('LR accuracy is:')
print(round(accuracy_score(y_cv,pred_cv)*100, 2),'%')


#(b)DECISION TREE ALGORITHM
#Fit model
from sklearn import tree
dt=tree.DecisionTreeClassifier(criterion='gini')
dt.fit(x_train,y_train)

#Predict values for cv data
pred_cv1=dt.predict(x_cv)
#Write test results in csv file
prediction_dt=pd.DataFrame({'Loan_ID':tmp,'predictions':pred_cv1}).reset_index(drop=True).to_csv('DecisionTreeClassifier_Predictions.csv')



#Evaluate accuracy of model
accuracy_score(y_cv,pred_cv1) #71.54%
matrix1=confusion_matrix(y_cv,pred_cv1)
print('DT accuracy is:')
print(round(accuracy_score(y_cv,pred_cv1)*100, 2),'%')


#(c)RANDOM FOREST ALGORITHM
#Fit model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)

#Predict values for cv data
pred_cv2=rf.predict(x_cv)
#Write test results in csv file
prediction_rf=pd.DataFrame({'Loan_ID':tmp,'predictions':pred_cv2}).reset_index(drop=True).to_csv('RandomForestClassifier_Predictions.csv')

#Evaluate accuracy of model
accuracy_score(y_cv,pred_cv2) #77.23%
matrix2=confusion_matrix(y_cv,pred_cv2)
print('Rand Forest accuracy is:')
print(round(accuracy_score(y_cv,pred_cv2)*100, 2),'%')


#(d)SUPPORT VECTOR MACHINE (SVM) ALGORITHM
from sklearn import svm
svm_model=svm.SVC()
svm_model.fit(x_train,y_train)

#Predict values for cv data
pred_cv3=svm_model.predict(x_cv)
#Write test results in csv file
prediction_svm=pd.DataFrame({'Loan_ID':tmp,'predictions':pred_cv3}).reset_index(drop=True).to_csv('SVM_Predictions.csv')


#Evaluate accuracy of model
accuracy_score(y_cv,pred_cv3) #64.23%
matrix3=confusion_matrix(y_cv,pred_cv3)
print('SVM accuracy is:')
print(round(accuracy_score(y_cv,pred_cv3)*100, 2),'%')


#(e)NAIVE BAYES ALGORITHM
from sklearn.naive_bayes import GaussianNB 
nb=GaussianNB()
nb.fit(x_train,y_train)

#Predict values for cv data
pred_cv4=nb.predict(x_cv)
#Write test results in csv file
prediction_nb=pd.DataFrame({'Loan_ID':tmp,'predictions':pred_cv4}).reset_index(drop=True).to_csv('GaussianNB.csv')


#Evaluate accuracy of model
accuracy_score(y_cv,pred_cv4) #80.49%
matrix4=confusion_matrix(y_cv,pred_cv4)
print('NBA accuracy is:')
print(round(accuracy_score(y_cv,pred_cv4)*100, 2),'%')


#(f)K-NEAREST NEIGHBOR(kNN) ALGORITHM
from sklearn.neighbors import KNeighborsClassifier
kNN=KNeighborsClassifier()
kNN.fit(x_train,y_train)

#Predict values for cv data
pred_cv5=kNN.predict(x_cv)
#Write test results in csv file
prediction_kNN=pd.DataFrame({'Loan_ID':tmp,'predictions':pred_cv5}).reset_index(drop=True).to_csv('KNeighborsClassifier.csv')

#Evaluate accuracy of model
accuracy_score(y_cv,pred_cv5) #64.23%
matrix5=confusion_matrix(y_cv,pred_cv5)
print('KNN accuracy is:')
print(round(accuracy_score(y_cv,pred_cv5)*100, 2),'%')


#(g) GRADIENT BOOSTING MACHINE ALGORITHM
from sklearn.ensemble import GradientBoostingClassifier
gbm=GradientBoostingClassifier()
gbm.fit(x_train,y_train)

#Predict values for cv data
pred_cv6=gbm.predict(x_cv)
#Write test results in csv file
prediction_gbm=pd.DataFrame({'Loan_ID':tmp,'predictions':pred_cv6}).reset_index(drop=True).to_csv('GradientBoostingClassifier.csv')


#Evaluate accuracy of model
accuracy_score(y_cv,pred_cv6) #78.86%
matrix6=confusion_matrix(y_cv,pred_cv6)
print('Gradient Boosting accuracy is:')
print(round(accuracy_score(y_cv,pred_cv6)*100, 2),'%')

#Select best model in order of accuracy
#Naive Bayes - 80.49%
#Logistic Regression - 78.86%
#Gradient Boosting Machine -78.86%
#Random Forest - 77.23%
#Decision Tree - 71.54%
#Support Vector Machine - 64.23%
#k-Nearest Neighbors(kNN) - 64.23%

#Predict values using test data (KNN)
pred_test=kNN.predict(x_cv)

#Write test results in csv file
predictions=pd.DataFrame(pred_test, columns=['predictions']).to_csv('Credit_Predictions.csv')

predictions=pd.DataFrame({'Loan_ID':tmp,'predictions':pred_test}).reset_index(drop=True)

#print(predictions)


