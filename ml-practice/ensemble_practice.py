import pandas as pd
from sklearn.model_selection import train_test_split

from XGBoost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#load the dataset
dataset = pd.read_csv("Finalized_V2.csv")

#specify the independent and target variables
x = dataset.iloc[:, 1:17].values
y = dataset.iloc[:, 17].values

#splitting the dataset into train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#importing the voting classifier
from sklearn.ensemble import VotingClassifier

model_1 = LogisticRegression()
model_2 = XGBClassifier()
model_3 =RandomForestClassifier()
model_4 = SVC() 

#combine three algorithms
final_model = VotingClassifier(estimators=[('lr', model_1), ('xgb', model_2), ('rf', model_3), ('svc', model_4)], voting='hard')

#model creation
 
final_model.fit(x_train, y_train) #train the algorithms

#check for evaluation
predict_final = final_model.predict(x_test)

from sklearn import metrics
print("Accuracy", metrics.accuracy_score(y_test, predict_final))