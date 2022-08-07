import pandas as pd

#load the dataset
dataset = pd.read_csv("Finalized_V2.csv")

#specify the independent and target variables
x = dataset.iloc[:, 1:17]
y = dataset.iloc[:, 17]

#splitting the dataset into train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#model creation
from XGBoost import XGBClassifier
model = XGBClassifier() 
model.fit(x_train, y_train)

#check for evaluation
xgb_predict = model.predict(x_test)

from sklearn import metrics
print("Accuracy", metrics.accuracy_score(y_test, xgb_predict))