import pandas as pd
import numpy as np

# # Tiny dataset engineered to contain missing values + mixed types
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

# print("HEAD:\n", df.head(), "\n") # head(n): first n rows,() return default 5 rows
# print('Head:\n', df0.head(), "\n")

# print("INFO:")                    #info gives Number of rows ;Number of columns;Column names;
#                                   #Non-null (non-missing) values;Data types of each column; Memory usage
# df.info()
# df0.info()

# print("\nDESCRIBE (numeric):")
# print(df.describe())             #count;mean;standard deviation;min;25% (Q1);50% (median);75% (Q3);max
# print(df0.describe())

# print("\nMISSING VALUES PER COLUMN (count):")
# print(df.isna().sum())           # returns True for every na, and false otherwise in a table format
# print(df0.isna())

# print("\nMISSING VALUES PER COLUMN (percent):")
# print((df.isna().mean() * 100).round(1))      #mean() finds the propotion of True values of each column
# print((df0.isna().mean() * 100).round(1))

# print("\nVALUE COUNTS: city (include NaN):")
# print(df["city"].value_counts(dropna=False)) # count the number of times a element appears in given column
# print(df0['city'].value_counts(dropna=False))

# print("\nVALUE COUNTS: member (include NaN):")
# print(df["member"].value_counts(dropna=False))
# print(df0["member"].value_counts()) # default is to not include na




print("ORIGINAL\n", df, "\n")
print("ORIGINAL\n", df0,'\n')

# A) Fill missing values
df_filled = df.fillna(
    {
        "city": "Unknown",
        "age": df["age"].median(),
        "spend": 0.0,
        "member": False,
    }
)

df_filled0 = df0.fillna(
    {
        'city' : 'Undiscovered',
        'age': df0['age'].mean(),
        'spend': 0.0,
        'member':False

    }

)
print("AFTER fillna(dict)\n", df_filled, "\n") 
print("AFTER fillna(dict)\n", df_filled0, "\n")  # fillna replaces NA/NaN values 

# B) Drop rows where critical columns are missing (start from original)
df_dropped = df.dropna(subset=["city", "age"])
print("AFTER dropna(subset=['city','age'])\n", df_dropped, "\n")  # dropna removes rows with NA 
df0_dropped = df0.dropna(subset=['city', 'age'])
print("AFTER dropna(subset=['city','age'])\n", df0_dropped, "\n") 

# C) Types: convert id to string, member to boolean-ish, age to integer (after filling)
df_typed = df_filled.copy()
df0_typed = df_filled0.copy()

df_typed["id"] = df_typed["id"].astype("string")
df_typed["member"] = df_typed["member"].astype("boolean")
df_typed["age_int"] = df_typed["age"].astype("int64")  # safe because we filled age first
df0_typed["id"] = df0_typed["id"].astype("float")
df0_typed['id_string'] = df0_typed["id"].astype('string')



print("AFTER astype\n")
print(df_typed)
print("\nDTYPES:\n", df_typed.dtypes)  # astype casts to a specified dtype 
print("AFTER astype\n")
print(df0_typed)
print("\nDTYPES:\n", df0_typed.dtypes)   #dtypes gives the data types of each column