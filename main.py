import numpy as np
import pandas as pd 

#Importing tools for visualization 
import matplotlib.pyplot as plt 
import seaborn as sns 
#Import evaluation metric librarie s
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report 
from sklearn.preprocessing import LabelEncoder
#Libraries used for data  prprocessing 
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

#Library used for ML Model implementation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB


#librries used for ignore warnings 
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("https://raw.githubusercontent.com/Apaulgithub/oibsip_task1/main/Iris.csv")
df.head()
df.tail()
df.describe()
print("Number of rows are: ", df.shape[0])
print("Number of columns are: ", df.shape[1])
#Dataset Info
#Checking information about the dataset using info
df.info()
#dataset duplicate value count
dup=df.duplicated().sum()
print(f'num of duplicated rows are {dup}')
#for finding null values
df.isnull().sum()
df.describe(include='all').round(2)
# Chart - 1 Histogram visualization code for distribution of numerical variables
# Create a figure with subplots
plt.figure(figsize=(8, 6))
plt.suptitle('Distribution of Iris Flower Measurements', fontsize=14)

# Create a 2x2 grid of subplots
plt.subplot(2, 2, 1)  # Subplot 1 (Top-Left)
plt.hist(df['SepalLengthCm'])
plt.title('Sepal Length Distribution')

plt.subplot(2, 2, 2)  # Subplot 2 (Top-Right)
plt.hist(df['SepalWidthCm'])
plt.title('Sepal Width Distribution')

plt.subplot(2, 2, 3)  # Subplot 3 (Bottom-Left)
plt.hist(df['PetalLengthCm'])
plt.title('Petal Length Distribution')

plt.subplot(2, 2, 4)  # Subplot 4 (Bottom-Right)
plt.hist(df['PetalWidthCm'])
plt.title('Petal Width Distribution')

# Display the subplots
plt.tight_layout()  # Helps in adjusting the layout
plt.show()

# Define colors for each species and the corresponding species labels.
colors = ['red', 'yellow','green']
species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# Chart - 2 Scatter plot visualization code for Sepal Length vs Sepal Width.
# Create a scatter plot for Sepal Length vs Sepal Width for each species.
for i in range(3):
    # Select data for the current species.
    x = df[df['Species'] == species[i]]

    # Create a scatter plot with the specified color and label for the current species.
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c=colors[i], label=species[i])

# Add labels to the x and y axes.
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Add a legend to identify species based on colors.
plt.legend()

# Display the scatter plot.
plt.show()


# Chart - 3 Scatter plot visualization code for Petal Length vs Petal Width.
# Create a scatter plot for Petal Length vs Petal Width for each species.
for i in range(3):
    # Select data for the current species.
    x = df[df['Species'] == species[i]]

    # Create a scatter plot with the specified color and label for the current species.
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])

# Add labels to the x and y axes.
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# Add a legend to identify species based on colors.
plt.legend()

# Display the scatter plot.
plt.show()

#encode the categorical column 
#create a tableEcoder object
le = LabelEncoder()

#Encode the 'species' column to convert the species name to numerical lables 
df['Species'] = le.fit_transform(df['Species'])

#check the unique value in the 'Species' column after encoding
unique_Species = df['Species'].unique()

#display the unique encoder 
print("Encoded Species value:")
print(unique_species) ,'iris-setosa' == 0, 'iris-versicolor'  == 1, 'iris-virginica' ==2


x = df.drop(columns=['Species'])
y = df['Species']

#Splitting the data to train and test
x_train,x_test,y_test,y_train=train_test_split(x,y,test_size=0.3)


#Checking the train distribution of dependent variable
y_train.value_counts()