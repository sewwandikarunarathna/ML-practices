#import required libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#load the dataset

dataset = pd.read_csv("Iris.csv")
#print(dataset)

#specify the independent and dependent variables
#here independent variable is x. dependent(target) variable is y.
#iloc is used to select particular columns and rows. [all rows, columns]. 
# :=all rows, 1:5= index of columns. in x, 1 is second column and upto 4th column
x = dataset.iloc[:, 1:5] 
y= dataset.iloc[:, 5]

#split the data into training and testing
#here training 80% and testing 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

#model creation 
#svc= support vector classifier
model = SVC() 
model.fit(x_train, y_train)




#check for evaluation
flower_predict = model.predict(x_test)

print("Accuracy", accuracy_score(y_test, flower_predict))