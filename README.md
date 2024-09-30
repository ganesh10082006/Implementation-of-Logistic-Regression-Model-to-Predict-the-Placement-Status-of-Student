# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy and confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GANESH G.
RegisterNumber:  212223230059
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
data = pd.read_csv("/Placement_Data.csv")
data1 = data.copy()
data1 = data1.drop(['sl_no','salary'], axis=1)
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
x = data1.iloc[:, :-1]
y = data1["status"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print("Accuracy Score:", accuracy)
print("\nConfusion Matrix:\n", confusion)
print("\nClassification Report:\n", cr)
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=model.classes_)
cm_display.plot()
```

## Output:

### Accuracy Score and Classification Report:
![image](https://github.com/user-attachments/assets/512820da-17a6-425a-ade4-40ce1df4488d)

### Displaying:
![image](https://github.com/user-attachments/assets/39ac254c-4c4d-466c-8361-353450021efa)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
