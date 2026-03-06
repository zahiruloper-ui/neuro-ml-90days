import pandas as pd
import numpy as np


df_str = pd.DataFrame({
    "name": ["Alice Smith", "Bob.Jones", "Carol-Lee", "David O'Connor"],
    "city": ["Vancouver, BC", "Toronto ON", "Calgary,AB", "Vancouver,BC"],
    "dept": ["Data Science", "ML Engineer", "Data Science", "ML Engineer"]
})
print(df_str)

print("Uppercase names: \n", df_str["name"].str.upper())
print("Length of names: \n", df_str["name"].str.len())
print("First 3 letters: \n", df_str["name"].str[:3])


# Extract first name (before space)
df_str["first_name"] = df_str["name"].str.extract(r'^(\w+)') # r'^(\w+)' , this is a regex pattern
print(df_str[["name", "first_name"]])

# Split name into first/last
names_split = df_str["name"].str.split(" ", expand=True) # str.split(" ") → splits on spaces
                                                         # expand=True → returns a DataFrame (one column per split part)
df_str["last_name"] = names_split[1].fillna(names_split[0])  # handle single names
print(df_str[["name", "first_name", "last_name"]])

#[aeiou] = any of these characters, .* = any characters (0+),case=False = ignore case, regex=False = literal string match
# Which names contain vowels?
has_vowels = df_str["name"].str.contains(r'[aeiou]', case=False, regex=True)
print("Has vowels:", has_vowels)

# Filter DataFrame
print("\nNames with vowels:")
print(df_str[has_vowels])


# Clean city names (remove comma + standardize province)
df_str["city_clean"] = df_str["city"].str.replace(r',.*', '', regex=True) #r',.*' strips everything after commas
print("Clean cities:", df_str["city_clean"])

# Standardize dept names
df_str["dept_short"] = df_str["dept"].str.replace(r'Data Science', 'DS', regex=False).str.replace(r'ML Engineer', 'MLE', regex=False)
print(df_str[["dept", "dept_short"]])

