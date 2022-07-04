Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
AIM:

To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Equipments Required:

    Hardware – PCs
    Anaconda – Python 3.7 Installation / Moodle-Code Runner

Algorithm

    Start the program
    Import pandas as pd
    insert and read the required csv file
    create the new variable data1 and copy the values to the new variable
    then check if the data1 is null and duplicated
    then import LabelEncoder from sklearn.preprocessing
    then create object for the LabelEncoder as le
    then using the le and fit_transform assign the unique integer
    then assign the column values for the x and y
    import train_test_split from sklearn.model_selection
    import Logistic Regression from sklearn.linear_model
    then import accuracy_score,confusion_matrix,classification_report from the sklearn.metrics
    finally print accuracy_score,confusion_matix,classification_report
    Stop the program

Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: p.vamsi reddy
RegisterNumber:  212220040110
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1.head()
x=data1.iloc[:,:-1]
y=data1["status"]
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear") #liblinear is nothing but library for larger linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

Output:

data.head image

image

image

image
Result:

Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
