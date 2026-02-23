import pandas as pd
import numpy as np

df = pd.DataFrame({
    "id": [101, 102, 103, 104, 105, 106, 107, 108],
    "city": ["Vancouver", "Toronto", "Vancouver", "Calgary", "Montreal", "Vancouver", "Toronto", "Calgary"],
    "score": [88, 72, np.nan, 95, 67, 88, 91, 72],
    "passed": [True, False, True, True, False, True, True, False],
    "dept": ["A", "B", "A", "C", "B", "A", "C", "B"],
})

print(df, "\n")



print(df[df['city'] == 'Vancouver'])            # used to find only rows with the specific column value
print(df.query("city == 'Vancouver' and passed == True")) # another way to do the same thing

print(df.query("city in ['Toronto', 'Vancouver'] and passed == True")) # for more than one item use in 


print(df.query("city == 'Vancouver'"))
print(df[df['city'].isin(['Vancouver', 'Toronto'])])   # inin() for more than one element

print(df[(df['dept']=='A') & (df['passed']==True)])

print(df[(df['city'].isin(['Vancouver', 'Toronto'])) & (df['score']>= 80)])

print(df[(df['city'] == 'Calgary') | (df['dept'] == 'C')])   # use | & in boolean mask 

print(df.query("city == 'Calgary' or dept == 'C'"))           # use or and in query


print(df[df['dept'].isin(['B', 'C'])])

print(df[(df['city'].isin(['Vancouver', 'Toronto'])) & 
         (df["passed"] == True) &
         (df['score'].notna())]) # use notna() to remove na


print(df['city'])   #df['colname'] produces all the elements of that column

print(df[['id','score']])  # df[['col1', 'col2']] to produce elements of > 1 columnn

print(df.loc[2:5,                       # both end points are selected in loc for splicing
             ['id', 'city', 'score']])  # loc[row, col] produces df with the specified rows and columns


print(df.iloc[2:5,                     # last end point is excluded like normal splicing in iloc
              [0, 1, 2]])              # index pos. is used for col and row instead of strings

print(df.loc[df["city"] == "Vancouver", ["id", "score"]])

print(df.loc[0:3, ["id", "city"]])
print(df.iloc[0:3, [0, 1]])


print(df.at[0, 'city'])   # at(rowno., colname) give the particular item in that col and row no.
print(df.iat[0, 1])       # iat uses index of col instead of str

df.loc[df['score'].isna(), 'score'] = 0   # putting value in place of na (editing)
print(df)


# sorting

print(df.sort_values(by='score'))    # sort_values(by = colname) sort according to ascending order (default)
print(df.sort_values(by='city'))      # works for both string(alphabetical A to Z), 
print(df.sort_values(by='passed'))    # int(ascending). and boolean (False to True) 

# For descending order, ascending = False

print(df.sort_values(by='score', ascending=False))    # sort_values(by = colname) sort according to ascending order (default)
print(df.sort_values(by='city', ascending=False))      # works for both string(alphabetical Z to A), 
print(df.sort_values(by='passed', ascending=False))   # int(descending). and boolean (True to False) 


# multiple sorting

print("\n multiple sorting: \n", df.sort_values(by=['city', 'score'], ascending=[True, False])) 


df_nan = df.copy()
df_nan.loc[2, "score"] = np.nan

print ("\n nan sorting \n", df_nan.sort_values(by= "score"))  #default is to put nan values is last
print (df_nan.sort_values(by= "score", na_position= "first"))
