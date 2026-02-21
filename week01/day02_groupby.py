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





out4 = (
    df.groupby("city")
      .agg(
          min_price=pd.NamedAgg(column="price", aggfunc="min"),
          max_price=pd.NamedAgg(column="price", aggfunc="max"),
          avg_price=pd.NamedAgg(column="price", aggfunc="mean"),
          total_qty=pd.NamedAgg(column="qty", aggfunc="sum"),
      )
      .reset_index()
)

#pd.NamedAgg allows you to choose column, choose agg, rename output column at the same time
# reset index makes the index column


out4a = (df1.groupby("city").agg(
    max_price = pd.NamedAgg(column='price', aggfunc='max'),
    min_price = pd.NamedAgg(column='price', aggfunc='min'),
    avg_price = pd.NamedAgg(column= 'price', aggfunc='mean'),
    total_qty = pd.NamedAgg(column='qty', aggfunc='sum'))
    
)


print(out4)

print(list(out4.columns))    # list() produces a list

print(out4a)



customers = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "name": ["Ana", "Bilal", "Chen"],
    "city": ["Vancouver", "Surrey", "Burnaby"],
})

customers1 = pd.DataFrame(
    {"cust_id" : [3, 6, 7],
     "name" : ['Nicky', 'Justin', 'Chen'],
     "city": ['Myanmar', 'China', 'Canada']
})

orders_2025 = pd.DataFrame({
    "order_id": [101, 102],
    "customer_id": [1, 2],
    "total": [35.5, 18.0],
})

orders_2025_1 = pd.DataFrame(
    {
        'order_id': [121, 10],
        'cust_id': [7, 6],
        'total': [7.89, 8.89]
    }
)

orders_2026 = pd.DataFrame({
    "order_id": [201, 202, 203],
    "customer_id": [2, 3, 999],  # 999 doesn't exist in customers (intentional)
    "total": [22.0, 44.0, 10.0],
})

orders_2026_1 = pd.DataFrame(
    {
        'order_id': [12, 10, 89],
        'cust_id': [3, 7, 999],
        'total': [78.9, 80.89, 90.0]
    }
)

# concat stacks rows (same columns)
orders_all = pd.concat([orders_2025, orders_2026], ignore_index=True)

#concat present all of the rows from two or more data frames\
# ignore_index = TRUE helps to make the index uniform,
# otherwise it will again start from 0 from next df

orders_all_1 = pd.concat([orders_2026_1, orders_2025_1], ignore_index=True)

# merge joins on keys (like SQL). Left join keeps all rows from orders_all.
orders_enriched = pd.merge(orders_all,           #left Df (keep all rows from this one)
                            customers,           #right Df (bring matching columns from this one)
                              on="customer_id",  # column used as the matching key in both tables
                                how="left")      # left join: keep all rows from orders_all,
                                                 # fill missing matches with NaN

orders_enriched1 = pd.merge(orders_all_1, customers1, on = 'cust_id'
                            , how = 'left')

print("orders_all shape:", orders_all.shape)  #shape tells the size of the table matrix (rows,cols)
print(orders_all, "\n")

print("orders_enriched shape:", orders_enriched.shape)
print(orders_enriched, "\n")

print(orders_all_1, "\n")
print(orders_enriched1)


