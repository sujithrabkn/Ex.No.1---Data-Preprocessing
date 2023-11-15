# Ex No1 Data Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:
Hardware – PCs

Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
Name: sujithra B K N
Reg NO: 212222230153
```
```
#importing libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:
#THE DATASET

![image](https://github.com/sujithrabkn/Ex.No.1---Data-Preprocessing/assets/119477857/675d7722-048d-485a-8de9-d51bb681d859)

#DROPPING UNWANTED FEATURE

![image](https://github.com/sujithrabkn/Ex.No.1---Data-Preprocessing/assets/119477857/1731c04d-c0c3-455b-a9d9-19ccf2391ec3)

#CHECKING FOR DUPLICATION

![image](https://github.com/sujithrabkn/Ex.No.1---Data-Preprocessing/assets/119477857/91964617-d312-4513-becf-36e3b181306f)

#DESCRIBING THE DATASET

![image](https://github.com/sujithrabkn/Ex.No.1---Data-Preprocessing/assets/119477857/b06c268e-cae6-4b2f-b833-a39a91cdf503)

#SCALING THE VALUES

![image](https://github.com/sujithrabkn/Ex.No.1---Data-Preprocessing/assets/119477857/43dfa9ab-e139-447d-8d64-0e4abb0f4329)

#X FEATURES

![image](https://github.com/sujithrabkn/Ex.No.1---Data-Preprocessing/assets/119477857/75ca96ba-4fb4-465f-a2c5-9d0e43656da3)

#Y FEATURES

![image](https://github.com/sujithrabkn/Ex.No.1---Data-Preprocessing/assets/119477857/a424febe-2d13-4e99-beae-66691e274c79)


#Splitting the training and testing dataset:

![image](https://github.com/sujithrabkn/Ex.No.1---Data-Preprocessing/assets/119477857/115bd770-1d61-48f5-ac8c-39e4ef884f57)



## RESULT
Thus we have successfully performed Data preprocessing in a data set downloaded from Kaggle.
