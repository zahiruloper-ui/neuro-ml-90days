import pandas as pd

df = pd.DataFrame({
    "city": ["Vancouver","Vancouver","Burnaby","Burnaby","Surrey","Surrey","Surrey","Richmond"],
    "category": ["A","B","A","B","A","A","B","B"],
    "price": [10, 12, 9, 14, 7, 8, 11, 13],
    "qty": [1, 2, 1, 3, 2, 4, 1, 2],
})

# 1) One column, many aggs (list)
out1 = df.groupby("city")["price"].agg(["min", "max", "mean"])

# 2) Many columns, different aggs (dict)
out2 = df.groupby("city").agg({"price": ["min", "max"], "qty": "sum"})

# 3) Multi-key group (Series result)
out3 = df.groupby(["city", "category"])["qty"].sum()

print("out1\n", out1, "\n")
print("out2\n", out2, "\n")
print("out3\n", out3, "\n")
