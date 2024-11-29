import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

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
rfC = RandomForestClassifier(random_state=100)
lrC = LogisticRegression(random_state=100)#max_iter=100
dtC = DecisionTreeClassifier(random_state=100)
nnC = MLPClassifier(random_state=100)
svC = svm.SVC(random_state=100)
knC = KNeighborsClassifier()
gnbC = GaussianNB()
gbC = GradientBoostingClassifier(random_state=100)

#this is where the random forest object is actually trained using training sets
rfC.fit(x_train, y_train)

#makes predictions on the x_train dataframe (phishing or legit)
y_rf_train_pred = rfC.predict(x_train)
#makes prediction on the x_test dataframe (phishing or legit)
y_rf_test_pred = rfC.predict(x_test)


lrC.fit(x_train, y_train)
y_lr_train_pred = lrC.predict(x_train)
y_lr_test_pred = lrC.predict(x_test)
accuracy_lr = accuracy_score(y_test, y_lr_test_pred)
precision_lr = precision_score(y_test, y_lr_test_pred)
recall_lr = recall_score(y_test, y_lr_test_pred)
conMatrix_lr = confusion_matrix(y_test, y_lr_test_pred)
classreport_lr = classification_report(y_test, y_lr_test_pred)
f1_score_lr = f1_score(y_test, y_lr_test_pred)

dtC.fit(x_train, y_train)
y_dtC_train_pred = dtC.predict(x_train)
y_dtC_test_pred = dtC.predict(x_test)
accuracy_dtC = accuracy_score(y_test, y_dtC_test_pred)
precision_dtC = precision_score(y_test, y_dtC_test_pred)
recall_dtC = recall_score(y_test, y_dtC_test_pred)
conMatrix_dtC = confusion_matrix(y_test, y_dtC_test_pred)
classreport_dtC = classification_report(y_test, y_dtC_test_pred)
f1_score_dtC = f1_score(y_test, y_dtC_test_pred)

nnC.fit(x_train, y_train)
y_nnC_train_pred = nnC.predict(x_train)
y_nnC_test_pred = nnC.predict(x_test)
accuracy_nnC = accuracy_score(y_test, y_nnC_test_pred)
precision_nnC = precision_score(y_test, y_nnC_test_pred)
recall_nnC = recall_score(y_test, y_nnC_test_pred)
conMatrix_nnC = confusion_matrix(y_test, y_nnC_test_pred)
classreport_nnC = classification_report(y_test, y_nnC_test_pred)
f1_score_nnC = f1_score(y_test, y_nnC_test_pred)

svC.fit(x_train, y_train)
y_svC_train_pred = svC.predict(x_train)
y_svC_test_pred = svC.predict(x_test)
accuracy_svC = accuracy_score(y_test, y_svC_test_pred)
precision_svC = precision_score(y_test, y_svC_test_pred)
recall_svC = recall_score(y_test, y_svC_test_pred)
conMatrix_svC = confusion_matrix(y_test, y_svC_test_pred)
classreport_svC = classification_report(y_test, y_svC_test_pred)
f1_score_svC = f1_score(y_test, y_svC_test_pred)

knC.fit(x_train, y_train)
y_knC_train_pred = knC.predict(x_train)
y_knC_test_pred = knC.predict(x_test)
accuracy_knC = accuracy_score(y_test, y_knC_test_pred)
precision_knC = precision_score(y_test, y_knC_test_pred)
recall_knC = recall_score(y_test, y_knC_test_pred)
conMatrix_knC = confusion_matrix(y_test, y_knC_test_pred)
classreport_knC = classification_report(y_test, y_knC_test_pred)
f1_score_knC = f1_score(y_test, y_knC_test_pred)

gnbC.fit(x_train, y_train)
y_gnbC_train_pred = gnbC.predict(x_train)
y_gnbC_test_pred = gnbC.predict(x_test)
accuracy_gnbC = accuracy_score(y_test, y_gnbC_test_pred)
precision_gnbC = precision_score(y_test, y_gnbC_test_pred)
recall_gnbC = recall_score(y_test, y_gnbC_test_pred)
conMatrix_gnbC = confusion_matrix(y_test, y_gnbC_test_pred)
classreport_gnbC = classification_report(y_test, y_gnbC_test_pred)
f1_score_gnbC = f1_score(y_test, y_gnbC_test_pred)

gbC.fit(x_train, y_train)
y_gbC_train_pred = gbC.predict(x_train)
y_gbC_test_pred = gbC.predict(x_test)
accuracy_gbC = accuracy_score(y_test, y_gbC_test_pred)
precision_gbC = precision_score(y_test, y_gbC_test_pred)
recall_gbC = recall_score(y_test, y_gbC_test_pred)
conMatrix_gbC = confusion_matrix(y_test, y_gbC_test_pred)
classreport_gbC = classification_report(y_test, y_gbC_test_pred)
f1_score_gbC = f1_score(y_test, y_gbC_test_pred)


#following metrics measure how well the predictions did against original labels
#accuracy is how many predictions were correct
accuracy_rf = accuracy_score(y_test, y_rf_test_pred)
#precision is how many of the samples flagged as phishing were actually phishing
precision_rf = precision_score(y_test, y_rf_test_pred)
#how well did identify all the phishing sites (number true phishing sites flagged / number of true phishing sites)
recall_rf = recall_score(y_test, y_rf_test_pred)
#combination of precision and recall, basically how good are the model's predictions
f1Score_rf = f1_score(y_test, y_rf_test_pred)
#gives 2x2 matrix for true positive, false negative
#                     false positive, true negative
conMatrix = confusion_matrix(y_test, y_rf_test_pred)
classReport = classification_report(y_test, y_rf_test_pred)

print("Random Forest Classifier")
print("Accuracy:  ", accuracy_rf)
print("Precision: ", precision_rf)
print("Recall:    ", recall_rf)
print("f1Score:   ", f1Score_rf)
print("")
print("Confusion Matrix:")
print(conMatrix)
print("")
#print("ClassReport")
#print(classReport)


print("Logistic Regression Classifier")
print("Accuracy:  ", accuracy_lr)
print("Precision: ", precision_lr)
print("Recall:    ", recall_lr)
print("f1Score:   ", f1_score_lr)
print("")
print("Confusion Matrix:")
print(conMatrix_lr)
print("")
#print("ClassReport")
#print(classReport)


print("Decision Tree Classifier")
print("Accuracy:  ", accuracy_dtC)
print("Precision: ", precision_dtC)
print("Recall:    ", recall_dtC)
print("f1Score:   ", f1_score_dtC)
print("")
print("Confusion Matrix:")
print(conMatrix_dtC)
print("")
#print("ClassReport")
#print(classReport)

print("Neural Network Classifier")
print("Accuracy:  ", accuracy_nnC)
print("Precision: ", precision_nnC)
print("Recall:    ", recall_nnC)
print("f1Score:   ", f1_score_nnC)
print("")
print("Confusion Matrix:")
print(conMatrix_nnC)
print("")
#print("ClassReport")
#print(classReport)

print("Support Vector Classifier")
print("Accuracy:  ", accuracy_svC)
print("Precision: ", precision_svC)
print("Recall:    ", recall_svC)
print("f1Score:   ", f1_score_svC)
print("")
print("Confusion Matrix:")
print(conMatrix_svC)
print("")
#print("ClassReport")
#print(classReport)

print("K-Neighbor Classifier")
print("Accuracy:  ", accuracy_knC)
print("Precision: ", precision_knC)
print("Recall:    ", recall_knC)
print("f1Score:   ", f1_score_knC)
print("")
print("Confusion Matrix:")
print(conMatrix_knC)
print("")
#print("ClassReport")
#print(classReport)

print("Gradient Boosting Classifier")
print("Accuracy:  ", accuracy_gbC)
print("Precision: ", precision_gbC)
print("Recall:    ", recall_gbC)
print("f1Score:   ", f1_score_gbC)
print("")
print("Confusion Matrix:")
print(conMatrix_gbC)
print("")
#print("ClassReport")
#print(classReport)

print("Gaussian Naive Bayes Classifier")
print("Accuracy:  ", accuracy_gnbC)
print("Precision: ", precision_gnbC)
print("Recall:    ", recall_gnbC)
print("f1Score:   ", f1_score_gnbC)
print("")
print("Confusion Matrix:")
print(conMatrix_gnbC)
print("")
#print("ClassReport")
#print(classReport)
