import pandas as pd

df = pd.read_csv("train.csv")

print(df.head())
print(df.shape)
print(df.columns)

df["Age"] = df["Age"].fillna(df["Age"].median())

print(df.isnull().sum())

print(df["Survived"].value_counts())

import matplotlib.pyplot as plt

survival_counts = df["Survived"].value_counts()

plt.bar(["Did Not Survive", "Survived"], survival_counts)

plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")
plt.title("Titanic Survival Count")

plt.savefig("survival_count.png")
plt.show()

gender_survival = df.groupby("Sex")["Survived"].sum()

plt.bar(gender_survival.index, gender_survival.values)

plt.xlabel("Gender")
plt.ylabel("Number of Survivors")
plt.title("Survival by Gender")

plt.savefig("gender_survival.png")

plt.show()

class_survival = df.groupby("Pclass")["Survived"].sum()

plt.bar(class_survival.index, class_survival.values)

plt.xlabel("Passenger Class")
plt.ylabel("Number of Survivors")
plt.title("Survival by Passenger Class")

plt.savefig("class_survival.png")
plt.show()

plt.hist(df["Age"], bins=20)

plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.title("Age Distribution of Titanic Passengers")

plt.savefig("age_distribution.png")
plt.show()