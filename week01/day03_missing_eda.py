import pandas as pd
import numpy as np

# Tiny dataset engineered to contain missing values + mixed types
df = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "city": ["Vancouver", "Vancouver", None, "Burnaby", "Burnaby"],
        "age": [29, np.nan, 41, 33, np.nan],      #np(numpy).nan(not a number)
        "spend": [120.5, 0.0, np.nan, 75.0, 10.0],
        "member": [True, True, False, None, False],
    }
)

df0 = pd.DataFrame({
    "id" : [4, 6, 8, 9, 10],
    'city' : ['Dhaka', 'Chittagong', 'Dhaka', 'Sylhet', None],
    'age' : [np.nan, 78, 89, 67, np.nan ],
    'spend' : [1123.0, 67.0, 56.56, np.nan, 10.0],
    'member' : [True, None, False, False, False]
}) 

print("HEAD:\n", df.head(), "\n") # head(n): first n rows,() return default 5 rows
print('Head:\n', df0.head(), "\n")

print("INFO:")                    #info gives Number of rows ;Number of columns;Column names;
                                  #Non-null (non-missing) values;Data types of each column; Memory usage
df.info()
df0.info()

print("\nDESCRIBE (numeric):")
print(df.describe())             #count;mean;standard deviation;min;25% (Q1);50% (median);75% (Q3);max
print(df0.describe())

print("\nMISSING VALUES PER COLUMN (count):")
print(df.isna().sum())           # returns True for every na, and false otherwise in a table format
print(df0.isna())

print("\nMISSING VALUES PER COLUMN (percent):")
print((df.isna().mean() * 100).round(1))      #mean() finds the propotion of True values of each column
print((df0.isna().mean() * 100).round(1))

print("\nVALUE COUNTS: city (include NaN):")
print(df["city"].value_counts(dropna=False)) # count the number of times a element appears in given column
print(df0['city'].value_counts(dropna=False))

print("\nVALUE COUNTS: member (include NaN):")
print(df["member"].value_counts(dropna=False))
print(df0["member"].value_counts()) # default is to not include na



