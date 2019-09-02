#1 Importing the libraries
import numpy as np
import pandas as pd
#JP
import matplotlib.pyplot as plt
#JP End
#2 Importing the data set
dataset = pd.read_csv('beer_data.csv')


#Printing first 10 rows of the dataset
print("\n----------------------------\n",dataset.head(10))


#3 Dealing with the categorical data

#spliting Cellar Temperature into Maximum and Minimum based on the given data and converting the type from str to int
dataset['Minimum_Cellar_Temp'] = dataset['Cellar Temperature'].apply(lambda x : int(x.split('-')[0].strip()))
dataset['Maximum_Cellar_Temp'] = dataset['Cellar Temperature'].apply(lambda x : int(x.split('-')[1].strip()))

#New dataset with selected features
dataset = dataset[['ABV', 'Ratings','Minimum_Cellar_Temp','Maximum_Cellar_Temp', 'Score']]

#Printing first 10 rows of the dataset
print("\n----------------------------\n",dataset.head(10))

#Printing the summary of the dataset
print("\n----------------------------\n")
print(dataset.info())


#4 Classifying dependent and independent variables

#All columns except the last column are independent features- (Selecting every column except Score)
X = dataset.iloc[:,:-1].values

#Only the last column is the dependent feature or the target variable(Score)
y = dataset.iloc[:,-1].values


#5 Creating training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25,random_state =1)

#JP Visualising the Data
X1 = dataset.iloc[:,0].values
X2 = dataset.iloc[:,1].values
X3 = dataset.iloc[:,2].values
plt.scatter(X1,y,color='red')
#plt.plot(X1,y,color='blue')
plt.title('Data')
plt.xlabel('X1')
plt.ylabel('y')
plt.show()

plt.scatter(X2,y,color='red')
#plt.plot(X1,y,color='blue')
plt.title('Data')
plt.xlabel('X2')
plt.ylabel('y')
plt.show()

plt.scatter(X3,y,color='red')
#plt.plot(X1,y,color='blue')
plt.title('Data')
plt.xlabel('X3')
plt.ylabel('y')
plt.show()

#JP Outlier from https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
from scipy import stats
z = np.abs(stats.zscore(dataset))
print(z)
#threshold = 3
#print(np.where(z > 6))

dataset = dataset[(z < 6).all(axis=1)]

#Printing the summary of the dataset
print("\n----------------------------\n")
print(dataset.info())

# jp End


#4 Classifying dependent and independent variables

#All columns except the last column are independent features- (Selecting every column except Score)
X = dataset.iloc[:,:-1].values

#Only the last column is the dependent feature or the target variable(Score)
y = dataset.iloc[:,-1].values


#5 Creating training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05,random_state =1)

#################Data Preprocessing Ends #################################


""" Multiple Linear regression """

#6 Creating the Regressor and training it with the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize = True)

#7 Feeding the data and training the model
regressor.fit(X_train,y_train)


#8 Predicting the Score for test set observations
y_pred = regressor.predict(X_test)

#printing the predictions
print("\n----------------------------\nPredictions = \n",y_pred)

#9 Calculating score from Root Mean Log Squared Error

def rmlse(y_test, y_pred):
    error = np.square(np.log10(y_pred +1) - np.log10(y_test +1)).mean() ** 0.5
    score = 1 - error
    return score

print("\n----------------------------\nRMLSE Score = ", rmlse(y_test, y_pred))