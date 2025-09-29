# Import the pandas library for data handling
import pandas as pd

# Define the URL of the Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# The CSV file doesn't have a header, so we'll define the column names manually
column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# Load the dataset from the URL into a pandas DataFrame
dataset = pd.read_csv(url, names=column_names)

# --- Display Summary and Statistics ---

# 1. Display the dimensions (shape) of the dataset
print("✅ Dataset Dimensions (Rows, Columns):")
print(dataset.shape)

# 2. Display a concise summary using info()
# This shows column data types, memory usage, and non-null value counts.
print("\n✅ Dataset Summary:")
dataset.info()

# 3. Display descriptive statistics using describe()
# This calculates the count, mean, standard deviation, min/max, and quartiles.
print("\n✅ Descriptive Statistics:")
print(dataset.describe())