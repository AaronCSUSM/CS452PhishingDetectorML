import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("https://raw.githubusercontent.com/vaibhavbichave/Phishing-URL-Detection/refs/heads/master/phishing.csv")
#print(df.head())
print(df.info())

df = df.drop(['Index'], axis=1)

#print(df.info())

y = df["class"]
#print(y.head())
x = df.drop('class', axis = 1)
#print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
x_train

rf = RandomForestClassifier(random_state=100)
rf.fit(x_train, y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)
#print(y_lr_train_pred)
#print(y_lr_test_pred)

accuracy = accuracy_score(y_test, y_rf_test_pred)
precision = precision_score(y_test, y_rf_test_pred)
recall = recall_score(y_test, y_rf_test_pred)
f1Score = f1_score(y_test, y_rf_test_pred)
conMatrix = confusion_matrix(y_test, y_rf_test_pred)
classReport = classification_report(y_test, y_rf_test_pred)


print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall", recall)
print("f1Score: ", f1Score)
print("ClassReport: ", classReport)
