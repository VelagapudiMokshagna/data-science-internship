import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("bank-full.csv", sep=";")

print(df.head())
print(df.shape)
print(df.columns)

print(df.isnull().sum())
print(df.dtypes)
print(df["y"].value_counts())

X = df.drop("y", axis=1)
y = df["y"]

X = pd.get_dummies(X)

print(X.shape)
print(X.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred[:10])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

import matplotlib.pyplot as plt

plt.bar(["Actual No", "Actual Yes"],
        [sum(y_test == "no"), sum(y_test == "yes")])

plt.xlabel("Purchase Outcome")
plt.ylabel("Number of Customers")
plt.title("Customer Purchase Outcome")

plt.savefig("purchase_outcome.png")
plt.show()