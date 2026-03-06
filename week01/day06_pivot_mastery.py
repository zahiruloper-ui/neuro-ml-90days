import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    "city": np.random.choice(["Vancouver","Toronto","Calgary"], 12),
    "dept": np.random.choice(["ML","DS"], 12),
    "score": np.random.randint(60, 100, 12)
})
print(df)

# pivot review
pivot_basic = df.pivot_table(values="score", index="city", columns="dept", aggfunc="mean")
print(pivot_basic)

# All with overall averages
pivot_margin = df.pivot_table(values="score", index="city", columns="dept", aggfunc="mean", margins=True)
print(pivot_margin)

# crosstab(defailt agg function is count)

ct_basic = pd.crosstab(df["city"], df["dept"])
print(ct_basic)


ct_margin = pd.crosstab(df["city"], df["dept"], margins=True)
print(ct_margin)


ct_norm = pd.crosstab(df["city"], df["dept"], normalize="index") # noramlize gives the proportion instead of 
                                                                 # count 
print(ct_norm)


# apply() basics

def get_grade(row):
    score = row["score"]
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"

# Apply to each row
df["grade"] = df.apply(get_grade, axis=1)
print(df[["city","dept","score","grade"]])

df["grade_lambda"] =df.apply(lambda row: "A" if row["score"] >= 90 else "B" if row["score"] >= 80
                              else "C" if row["score"] >= 70 else "F", axis=1)

print(df[["score","grade","grade_lambda"]].tail())   # tail() only shows the last 5 rows
 


# apply() advanced
def city_dept_grade(row):
    if row["score"] >= 85 and row["city"] == "Calgary":
        return "Top Calgary"
    elif row["score"] >= 90:
        return "A-Student"
    else:
        return row["grade"]  # fallback

df["status"] = df.apply(city_dept_grade, axis=1)
print(df[["city","score","grade","status"]].head(6))





