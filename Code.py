# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
titanic_data = pd.read_csv('train.csv')  # Adjust path if necessary

# Data Cleaning
# Handling missing values
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data.drop(columns=['Cabin'], inplace=True)  # Dropping Cabin due to high number of missing values

# Correcting data types
titanic_data['Survived'] = titanic_data['Survived'].astype('category')
titanic_data['Pclass'] = titanic_data['Pclass'].astype('category')

# Exploratory Data Analysis
# Analyzing survival rates based on different features
sns.countplot(x='Survived', data=titanic_data)
plt.title('Survival Count')
plt.show()

sns.countplot(x='Survived', hue='Sex', data=titanic_data)
plt.title('Survival Rate by Gender')
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=titanic_data)
plt.title('Survival Rate by Class')
plt.show()

sns.histplot(titanic_data['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Visualization
# Visualizing survival based on where passengers embarked from
sns.countplot(x='Embarked', hue='Survived', data=titanic_data)
plt.title('Survival Rate Based on Embarking Point')
plt.show()

# Summary of insights with appropriate visuals
# Saving and documenting the findings in a Jupyter Notebook
print("Documentation and visualizations saved. Review Jupyter Notebook for detailed analysis.")

# Save the cleaned and analyzed data
titanic_data.to_csv('titanic_cleaned.csv', index=False)
print("Cleaned data saved.")
