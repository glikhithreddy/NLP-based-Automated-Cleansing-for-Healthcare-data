import csv

# Sample data
data = [
    ["CustomerID", "Name", "Email", "JoinDate", "AmountSpent"],
    [1, "John Doe", "john@example.com", "2024-01-15", 150.00],
    [2, "Jane Smith", "jane@example.com", "2024-02-20", 200.00],
    [3, "Bob Johnson", "", "2024-03-05", 150.00],
    [4, "Mary Johnson", "mary@example.com", "2024-02-30", 300.00],  # Corrected invalid date
    [5, "Tom Brown", "tom@example.com", "2024-03-15", 400.00],
    [6, "Emily Davis", "emily@example.com", "2024-01-25", ""],
    # Duplicate row removed (assuming it's an error)
]

# Open the file in write mode with newline='' to avoid potential newline issues
with open("business_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

print("CSV file 'business_data.csv' created successfully!")


import pandas as pd

data = pd.read_csv('business_data.csv')

print(data.head())

print(data.describe())

print(data.isnull().sum())

print(data.dtypes)

print(data['CustomerID'].unique())


import re


# Function to validate email
def is_valid_email(email):
 pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
 return bool(re.match(pattern, email))

# Validate emails
data['Email_Valid'] = data['Email'].apply(lambda x: is_valid_email(x) if
pd.notnull(x) else False)

print(data[['Email', 'Email_Valid']])


# Check completeness
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Check for duplicates
duplicates = data.duplicated().sum()
print("Number of duplicate entries:", duplicates)

# Display duplicates
print(data[data.duplicated()])

# Check for valid JoinDate format
data['JoinDate'] = pd.to_datetime(data['JoinDate'], errors='coerce')
invalid_dates = data[data['JoinDate'].isnull()]
print("Invalid Join Dates:\n", invalid_dates)

# Display columns to evaluate relevance
print("Columns in the dataset:\n", data.columns)

# Check for unique CustomerID
unique_ids = data['CustomerID'].nunique()
print("Unique Customer IDs:", unique_ids)

# Fill missing AmountSpent with mean
mean_amount = data['AmountSpent'].mean()
data['AmountSpent'].fillna(mean_amount, inplace=True)

# Remove duplicates
data.drop_duplicates(inplace=True)

# Remove rows with invalid emails
data = data[data['Email_Valid']]

# Remove rows with invalid JoinDates
data = data[data['JoinDate'].notnull()]

# Check for missing values again
print("Missing Values after cleaning:\n", data.isnull().sum())

# Check for duplicates again
print("Number of duplicate entries after cleaning:", data.duplicated().sum())

with open('data_quality_report.txt', 'w') as report:
 report.write("Data Quality Assessment Report\n")
 report.write("=================================\n")
 report.write(f"Total Rows: {len(data)}\n")
 report.write(f"Missing Values: {data.isnull().sum().to_dict()}\n")
 report.write(f"Duplicate Entries: {data.duplicated().sum()}\n")
 report.write(f"Invalid Emails: {data[data['Email_Valid'] == False].shape[0]}\n")
 report.write(f"Invalid Join Dates: {data[data['JoinDate'].isnull()].shape[0]}\n")

