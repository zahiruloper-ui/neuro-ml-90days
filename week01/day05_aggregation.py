
import pandas as pd

df = pd.DataFrame({
    "name": ["Ava","Ben","Cara","Dan"],
    "city": ["Vancouver","Toronto","Vancouver","Calgary"],
    "score": [88, 72, 88, 91],
    "dept": ["ML","DS","DS","ML"]
})



# use of group_by

mean_by_city = df.groupby("city")["score"].mean() # Groups by unique values in "city".

print(mean_by_city)


# use of aggregation ag

city_stats = df.groupby("city")["score"].agg(['count', 'mean', 'min', 'max']) # allows multiple stat calc.
print(city_stats)

#as_index=False (keeps city as a regular column)

mean_by_city_index= df.groupby("city", as_index=False)["score"].mean() # creates a index column
print(mean_by_city_index)

mean_by_city_index_different_way = df.groupby("city")["score"].mean().reset_index()
print(mean_by_city_index_different_way)


# multi-group sorting
mean_multi_group = df.groupby(["city", "dept"])["score"].mean()
print(mean_multi_group)

sorted_multi = mean_multi_group.sort_values(ascending=False)
print(sorted_multi)

sorted_multi_indexing = sorted_multi.reset_index()
print(sorted_multi_indexing)


# pivot table (kind of like contingency table)

pivot_table = df.pivot_table(values= 'score', index = 'city', columns='dept', aggfunc= 'mean')
print(pivot_table)

# index= → rows, columns= → columns, values= → what to aggregate, 
# Combos that don't exist become NaN 

# ways to fill up the missing NaN values

pivot_filled = df.pivot_table(values= 'score', index = 'city', columns='dept', aggfunc= 'mean', fill_value= 0)
print(pivot_filled)
