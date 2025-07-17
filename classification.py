#Import necessary packages
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 

%matplotlib inline

# Assigning column names and reading the dataset
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Class_labels']
df = pd.read_csv('iris.data', names=columns)
df.head(-10) # Display the first 5 and last 5 rows of the dataset

# Check for missing values in the dataset
df.isnull().sum()

# Dataset Duplicate Value Count
dup = df.duplicated().sum()
print(f'number of duplicated rows are {dup}')

duplicates_all = df[df.duplicated(keep=False)]
print(duplicates_all)
df.describe() # statistical summary of the dataset
sns.pairplot(df, hue='Class_labels') # visualizing whole dataset
data = df.values # Convert DataFrame to NumPy array
X = data[:,0:4] # taking all rows and first 4 columns
# X contains the features (sepal length, sepal width, petal length, petal width)

Y = data[:,4] # taking all rows and last column
# Y contains the class labels (species of iris flower)
# Calculate the average of each feature for each class label using list comprehension
Y_Data = np.array(
    [np.average(X[:, i][Y==j].astype('float32')) 
     for i in range (X.shape[1]) 
     for j in (np.unique(Y))
]
)

# Reshape the Y_Data to match the number of classes and features
Y_Data_reshaped = Y_Data.reshape(4, 3)

# Transpose the array to have classes as rows and features as columns
Y_Data_reshaped = np.swapaxes   (Y_Data_reshaped, 0, 1) 

# set x-axis and bar width
X_axis = np.arange(len(columns)-1)
width = 0.25
plt.bar(X_axis, Y_Data_reshaped[0], width, label='Setosa')
plt.bar(X_axis + width, Y_Data_reshaped[1], width, label='Versicolor')

plt.bar(X_axis + 2*width, Y_Data_reshaped[2], width, label='Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Support vector machine classifier
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train) # training the model wtih default parameters

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions, labels=np.unique(y_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

X_new = np.array([
    [3, 2, 1, 0.2], 
    [4.9, 2.2, 3.8, 1.1 ], 
    [5.3, 2.5, 4.6, 1.9 ],
    [5.1, 3.5, 1.4, 0.2]
    ])
#Prediction of the species from the input vector, each row is a new sample
prediction = model.predict(X_new)
print("Prediction of Species: {}".format(prediction))

import pickle

with open('IrisModel.pickle', 'wb') as f:
    pickle.dump(model, f)

# load the model from the file
with open('IrisModel.pickle', 'rb') as f:
    loaded_model = pickle.load(f)

model.predict(X_new)  
loaded_model.predict(X_new)  # Verify that the loaded model can make predictions


what is the difference between model and loaded_model in context of this jupyter file
