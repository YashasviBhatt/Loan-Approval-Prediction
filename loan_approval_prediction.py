# Importing important Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing Data Preprocessing Libraries
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importing the Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Importing Dataset
df = pd.read_csv('data.csv')

#----------------------------------------Data Preprocessing and Data Cleaning----------------------------------------

# Storing All Column Names in a List
col_names = list(df.columns)
col_names.remove('Loan_ID')

'''
# Printing all the Columns with Number of Missing Values present
print('Number of Missing Values per Column')
for col in col_names:
    if df[col].isnull().sum() > 0:
        print(f'{col} : {df[col].isnull().sum()}')
'''

'''
# Printing the Categories present in the respective Categorical Features
for col in col_names:
    if df[col].dtype == 'O':
        print(f'{col} : {list(dict(df[col].value_counts()).keys())}')
'''

# Setting up the Encoding Criteria for different Columns
enc = {
    'Gender' : {'Male' : 0, 'Female' : 1},
    'Married' : {'Yes' : 0, 'No' : 1},
    'Dependents' : {'0' : 0, '1' : 1, '2' : 2, '3+' : 3},
    'Education' : {'Graduate' : 0, 'Not Graduate' : 1},
    'Self_Employed' : {'No' : 0, 'Yes' : 1},
    'Property_Area' : {'Semiurban' : 0, 'Urban' : 1, 'Rural' : 2},
    'Loan_Status' : {'Y' : 0, 'N' : 1}
}

# Replacing Categorical Values with the Encoded Values
for col in col_names:
    if df[col].dtype == 'O':
        df[col] = df[col].replace(enc[col])

# Creating Separate Dataframes for Features and Class
X = df.iloc[:, :-1].values
y = df.iloc[:, 12].values

# Removing Loan_ID Column from the Dataset
X = np.delete(X, 0, 1)

# Creating Instances of Imputer Class for Missing Value Management
imputer_mode = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imputer_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)

# Replacing 'NaN' Values with Mode of the Values in the Respective Columns
X[:7] = imputer_mode.fit_transform(X[:7])
X[9:] = imputer_mode.fit_transform(X[9:])

# Replacing 'NaN' Values present in "LoanAmount" Column with the Mean of the Values of that Column
X_temp = X[:, 7].reshape(-1, 1)
X_temp = imputer_mean.fit_transform(X_temp)
X_temp = X_temp.reshape(1, -1)
X = np.delete(X, 7, 1)
X = np.insert(X, 7, X_temp, axis=1)

#----------------------------------------Data Preprocessing and Data Cleaning----------------------------------------

#----------------------------------------Model Building and Training----------------------------------------

# Splitting Dataframes into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Creating a Classifier List and Empty List for Model Evaluation
classifiers = ['Decision Tree Classifier', 'Logistic Regression', 'K Nearest Neighbors Classifier', 'Random Forest Classifier']
scores = list()

# Training using Decision Tree Classifier
clf1 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)

# Training using Logistic Regression
clf2 = LogisticRegression()
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)

# Training using K Nearest Neighbor Classifier
clf3 = KNeighborsClassifier(n_neighbors=7, metric='minkowski')
clf3.fit(X_train, y_train)
y_pred = clf3.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)

# Training using Random Forest Classifier
clf4 = RandomForestClassifier(n_estimators = 20)                     # 20 Estimator Decision Trees
clf4.fit(X_train, y_train)
y_pred = clf4.predict(X_test)
score = accuracy_score(y_test, y_pred)
scores.append(score)

#----------------------------------------Model Building and Training----------------------------------------

#----------------------------------------Model Evaluation----------------------------------------

# Plotting a Bar Plot to Evaluate the Performance of different Classifiers
sns.barplot(y=classifiers, x=scores)
plt.xlabel('Accuracy Score')
plt.ylabel('Classifier')
plt.title('Classifier Performance')
plt.show()

# As we can see that Random Forest Classifier has the best Accuracy Score, therefore we'll use it as the Final Model

#----------------------------------------Model Evaluation----------------------------------------

# Checking on Sample Data
ds = [
    ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'],
    ['Male', 'No', '0', 'Graduate', 'No', 5849, 0, 8000, 360, 1, 'Urban']
]

ds = pd.DataFrame(ds[1:], columns=ds[0])

for col in col_names[:-1]:
    if ds[col].dtype == 'O':
        ds[col] = ds[col].replace(enc[col])

[print('Yes') if clf4.predict(ds)[0] == 0 else print('No')]