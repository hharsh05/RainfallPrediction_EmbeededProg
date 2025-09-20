import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Display initial data information
print("Initial Data Info:")
print(df.info())

# 1. Handling Missing Values
# Display missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Option 1: Drop rows with missing values
df.dropna(inplace=True)

# Option 2: Fill missing values (example: fill with mean for numerical columns)
# df.fillna(df.mean(), inplace=True)

# 2. Removing Duplicates
df.drop_duplicates(inplace=True)

# 3. Data Type Conversion
# Convert a column to a specific data type (example: 'date_column' to datetime)
df['date_column'] = pd.to_datetime(df['date_column'], errors='coerce')

# 4. Renaming Columns
df.rename(columns={'old_name': 'new_name'}, inplace=True)

# 5. Removing Unwanted Characters
# Example: Remove special characters from a string column
df['text_column'] = df['text_column'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

# 6. Normalizing Data
# Example: Convert text to lowercase
df['text_column'] = df['text_column'].str.lower()

# 7. Outlier Detection and Removal
# Example: Remove outliers based on z-score
from scipy import stats
df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

# 8. Removing Unuseful Columns
# Specify the columns to drop
columns_to_drop = ['unuseful_column1', 'unuseful_column2']  # Replace with actual column names
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# 9. Final Data Check
print("\nCleaned Data Info:")
print(df.info())

# Save the cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)
