# Import pandas
import pandas as pd
import time
start_time = time.time()
# Read in dataset
transfusion = pd.read_csv('transfusion_LaoCai_final.csv')

# Print out the first rows of our dataset
transfusion.head()

transfusion.info()

# Rename target column as 'target' for brevity 
transfusion.rename(
    columns={'whether he/she donated blood in Februry 2022': 'target'},
    inplace=True
)
# Print out the first 2 rows
print("-------------")
#define function to calculate cv
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 

#calculate CV

print(1.96*transfusion.sem())
print("-------------")
# Print target incidence proportions, rounding output to 3 decimal places
print(transfusion.target.value_counts(normalize=True).round(3))

# Import train_test_split method
from sklearn.model_selection import train_test_split

# Split transfusion DataFrame into
# X_train, X_test, y_train and y_test datasets,
# stratifying on the `target` column
X_train, X_test, y_train, y_test = train_test_split(
    transfusion.drop(columns='target'),
    transfusion.target,
    test_size=0.2,
    random_state=42,
    stratify=transfusion.target
)

from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(tol=0.0001, C=1.0,solver='liblinear', max_iter = 100)

# fit the model with data
logreg.fit(X_train,y_train)

print(logreg.feature_names_in_    )
#
start_time = time.time()
y_pred=logreg.predict(X_test)

print("--- %s seconds ---" % (time.time() - start_time))


y_pred=logreg.predict(X_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


##class_names=[0,1] # name  of classes
##fig, ax = plt.subplots()
##tick_marks = np.arange(len(class_names))
##plt.xticks(tick_marks, class_names)
##plt.yticks(tick_marks, class_names)
### create heatmap
##sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
##ax.xaxis.set_label_position("top")
##plt.tight_layout()
##plt.title('Confusion matrix', y=1.1)
##plt.ylabel('Actual label')
##plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
##
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)


plt.show()
