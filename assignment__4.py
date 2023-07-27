import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset with the help of the archive url
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
df = pd.read_csv(url, names=names)

# Printing the first few rows of the datasets
print(df.head())

# Summary statistics of the dataset
print(df.describe())

# Check the data types and missing values in each column
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Check the number of instances for each class
print(df["num"].value_counts())

# Pairplot to visualize relationships between features
sns.pairplot(df, hue="num")
plt.show()

# Boxplot to visualize the distribution of each feature for each class
plt.figure(figsize=(12, 10))
for i, feature in enumerate(names[:-1]):
    plt.subplot(4, 4, i+1)
    sns.boxplot(x="num", y=feature, data=df)
plt.show()

# Correlation heatmap to check the correlation between features
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
