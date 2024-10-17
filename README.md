# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
STEP 1: Start

STEP 2: Load the employee churn dataset into a Pandas DataFrame and inspect the first few rows using data.head().

STEP 3: Check for missing values in the dataset using data.isnull().sum() and view the structure using data.info().

STEP 4: Preprocess the categorical data. Use LabelEncoder to convert the "salary" column into numerical values.

STEP 5: Define the feature matrix (X) by selecting relevant columns (e.g., satisfaction_level, last_evaluation, etc.) and set the target variable (Y) as the "left" column.

STEP 6: Split the dataset into training and testing sets using train_test_split() with a test size of 20%.

STEP 7: Initialize the Decision Tree Classifier with the entropy criterion and fit the model to the training data (x_train, y_train).

STEP 8: Predict the target values on the testing set (x_test) using dt.predict().

STEP 9: Calculate the accuracy of the model using metrics.accuracy_score() by comparing predicted values with actual test values.

STEP 10: Use the trained model to predict employee churn for a new input using dt.predict([[...]]).

STEP 11: End
```
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: APARNA RB
RegisterNumber: 212222220005 
*/

import pandas as pd

data=pd.read_csv("C:/Users/LENOVO/Downloads/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import  metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![369267409-9231c221-b787-409a-b110-1e7a98f943ce](https://github.com/user-attachments/assets/ce45a456-c8ec-4ac1-94ea-dc35cc313fd5)
![369267493-5b9ae4b1-f5ed-4ea8-802e-8a536c39d06b](https://github.com/user-attachments/assets/6d258501-a4c8-491c-af01-be6016779cbe)
![369267676-fa01c66a-64e2-47a4-830d-4dc144a82672](https://github.com/user-attachments/assets/7fe2c678-93d6-45cc-8ae5-db493d35f7ff)
![369267745-44552d9d-d2d4-4ee0-a11d-6aba116ca385](https://github.com/user-attachments/assets/1b625d3a-a891-43c2-8b81-6673a7c6238c)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
