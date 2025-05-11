#                  ****must install to use*****
#                  pip install ucimlrepo

import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
spambase = fetch_ucirepo(id=94) 
  
#data (as pandas dataframes) 
X = spambase.data.features 
y = spambase.data.targets 

#data reduction
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(0.14)
X_reduction = selector.fit_transform(X)

#split using 20/80 test train split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_reduction, y, test_size = 0.20, random_state = 0)

#scale data using the standard scalar
from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.fit_transform(X_test)

#creating Naïve Bayes classifier and predicting values
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train.values.ravel())
 
y_pred_bayes = classifier.predict(X_test) #bayes prediction

#creating Random Forest classifier and predicting values
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 21, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train.values.ravel())
 
y_pred_forest = classifier.predict(X_test) #forest predictions


#the accuracy, precision, recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score 

#Bayes Metrics
print("\n---------Naïve Bayes Classifier Metrics---------")

#Calculating the accuracy
accuracy_bayes = accuracy_score (Y_test, y_pred_bayes)
print("\nThe accuracy is:", accuracy_bayes * 100 ,"%")
 
#Claculating the recall
recall_bayes = recall_score (Y_test, y_pred_bayes)
print("The recall is:", recall_bayes * 100 ,"%")
#Calculating the precision
precision_bayes = precision_score(Y_test, y_pred_forest)
print("The precision is: ", precision_bayes * 100 , "%")


#Forest Metrics
print("\n---------Random Forest Classifier Metrics---------")

#Calculating the accuracy
accuracy_forest = accuracy_score (Y_test, y_pred_forest)
print("\nThe accuracy is:", accuracy_forest * 100 ,"%")
 
#Claculating the recall
recall_forest = recall_score (Y_test, y_pred_forest)
print("The recall is:", recall_forest * 100 ,"%")

#Calculating the precision
precision_forest = precision_score(Y_test, y_pred_forest)
print("The precision is: ", precision_forest * 100 , "%")


#ROC curve
from sklearn.metrics import roc_curve, auc 

fpr_bayes, tpr_bayes, threshold = roc_curve(Y_test, y_pred_bayes) #Bayes Curve
fpr_forest, tpr_forest, threshold = roc_curve(Y_test, y_pred_forest) #Forest Curve

roc_auc_bayes = auc(fpr_bayes, tpr_bayes) #Bayes Curve
roc_auc_forest = auc(fpr_forest, tpr_forest) #Forest Curve


plt.plot(fpr_bayes, tpr_bayes, 'b', label = 'AUC Bayes = %0.2f'  % roc_auc_bayes) #plot Bayes
plt.plot(fpr_forest, tpr_forest, 'g', label = 'AUC Forest= %0.2f'  % roc_auc_forest) #plot Forest

#Formatfor both plots
plt.title('ROC Curve')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#bar graph to compare between the accuracy for each classifier
plt.bar(['Naive Bayes', 'Random Forest'], [accuracy_bayes,accuracy_forest], color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy Comparison')
plt.ylim(0, 1)
plt.show()
