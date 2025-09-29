# Step 1: Import the pandas library
# We import pandas, a powerful library for data manipulation and analysis.
# 'pd' is the standard alias used by the community.
import pandas as pd

# Step 2: Load the dataset
# We provide the URL to the raw dataset file.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Since the file doesn't have a header row, we define the column names.
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# We use pandas' read_csv() function to load the data from the URL into a DataFrame.
# A DataFrame is a 2D table-like data structure.
dataset = pd.read_csv(url, names=column_names)

# Step 3: View the dataset
print("Successfully loaded the dataset!")

# Display the first 5 rows to get a quick look at the data.
print("\nFirst 5 rows of the dataset:")
print(dataset.head())

# Display a concise summary of the DataFrame.
# This includes the data type of each column and the number of non-null values.
print("\nDataset Information:")
dataset.info()

# Display basic statistical details (count, mean, std, etc.) for numerical columns.
print("\nStatistical Summary:")
print(dataset.describe())