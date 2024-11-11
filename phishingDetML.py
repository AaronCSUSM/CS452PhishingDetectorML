import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#use pandas to read in the dataset from url into a dataframe
df = pd.read_csv("https://raw.githubusercontent.com/vaibhavbichave/Phishing-URL-Detection/refs/heads/master/phishing.csv")
#.info() prints out basic information about the dataset such as number of samples, and number and names of columns
#print(df.info())

#drops the index column from the dataset. Dropping this increased the accuracy by about 1%
df = df.drop(['Index'], axis=1)

#saves the last column, class, which labels each sample as phishing (1) or legitimate (-1) into a seperate dataframe
y = df["class"]
#saves the rest of the columns into the x dataframe. This is all our x/feature variables for training the ML model
x = df.drop('class', axis = 1)

#splits data into training and testing sets
#(input variables, labels, portion reserved for testing, any number to control randomness)
#without the last variable the values change each time its run
#x_train and x_test are dataframes, y_train and y_test are series (single column)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
#print(x_train)

#create a random forest classifier object
#random_state parameter serves same purpose as it did for the train_test_split function. 
#Results change each time without it
rf = RandomForestClassifier(random_state=100)
#this is where the random forest object is actually trained using training sets
rf.fit(x_train, y_train)

#makes predictions on the x_train dataframe (phishing or legit)
y_rf_train_pred = rf.predict(x_train)
#makes prediction on the x_test dataframe (phishing or legit)
y_rf_test_pred = rf.predict(x_test)

#following metrics measure how well the predictions did against original labels
#accuracy is how many predictions were correct
accuracy = accuracy_score(y_test, y_rf_test_pred)
#precision is how many of the samples flagged as phishing were actually phishing
precision = precision_score(y_test, y_rf_test_pred)
#how well did identify all the phishing sites (number true phishing sites flagged / number of true phishing sites)
recall = recall_score(y_test, y_rf_test_pred)
#combination of precision and recall, basically how good are the model's predictions
f1Score = f1_score(y_test, y_rf_test_pred)
#gives 2x2 matrix for true positive, false negative
#                     false positive, true negative
conMatrix = confusion_matrix(y_test, y_rf_test_pred)
classReport = classification_report(y_test, y_rf_test_pred)


print("Accuracy:  ", accuracy)
print("Precision: ", precision)
print("Recall:    ", recall)
print("f1Score:   ", f1Score)
print("")
print("Confusion Matrix:")
print(conMatrix)
print("")
#print("ClassReport")
#print(classReport)
