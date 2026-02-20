import pandas as pd

df = pd.DataFrame({
    "city": ["Vancouver","Vancouver","Burnaby","Burnaby","Surrey","Surrey","Surrey","Richmond"],
    "category": ["A","B","A","B","A","A","B","B"],
    "price": [10, 12, 9, 14, 7, 8, 11, 13],
    "qty": [1, 2, 1, 3, 2, 4, 1, 2],  
})                             # this is a dictionary  has to be inside {}
                               # starts with a column name "" :
                               # then a list of column values inside [] 


df1 = pd.DataFrame({
    "city" : ['Lisbon', 'Dhaka', 'Lahore'],
    "cat" : ['A', 'C', 'D'],
    'price' : [10, 34, 55],
    'qty' : [3, 6, 7]
 })

# 1) One column, many aggs (list)
out1 = df.groupby("city")["price"].agg(["min", "max", "mean"])  
 # agg (short for aggregate) is a Pandas function used to apply one or more
 # summary functions to data.

out1a = df1.groupby("cat")["qty"].agg(["min", "max", "mean"])
# groupby Split the DataFrame into groups based on the (column)
# [column] only look at this column to perform the functions given in agg[]
# There are more functions like min, max and mean


# 2) Many columns, different aggs (dict)
out2 = df.groupby("city").agg({"price": ["min", "max"], "qty": "sum"})

out2a = df1.groupby("city").agg({"price": ["min", "max"], "qty" : ["sum", "std"]})
# to use multiple columns in calc. use format like dict

# 3) Multi-key group (Series result)
out3 = df.groupby(["city", "category"])["qty"].sum()

out3a = df1.groupby(["city", "cat"])["qty"].agg(["sum", "min"])

# use [] when working more than one item
# use () after a function
# use {} in making a dictonary



print("out1\n", out1, "\n")
print("out2\n", out2, "\n")
print("out3\n", out3, "\n")
print("out1a\n", out1a, "\n")
print("out2a\n", out2a, "\n")
print("out3a\n", out3a, "\n")