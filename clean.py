import pandas as pd

data = pd.read_csv("titanic.csv")

# 1. Remove repeated rows (there are none here, but a good habit)
data = data.drop_duplicates()

# 2. Fill empty ages with the middle age
data["Age"] = data["Age"].fillna(data["Age"].median())

# 3. Fill empty cabins with the word Unknown
data["Cabin"] = data["Cabin"].fillna("Unknown")

# 4. Fill the 2 empty boarding ports with the most common one
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# 5. Check that no empty boxes are left
print(data.isna().sum())

# 6. Save the clean file
data.to_csv("clean_titanic.csv", index=False)