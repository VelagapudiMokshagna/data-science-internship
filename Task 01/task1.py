import pandas as pd

df = pd.read_csv("population.csv", skiprows=4)

df = df.drop(columns=["Unnamed: 69"])

print(df.shape)

import matplotlib.pyplot as plt

countries = [
    "India", "China", "United States", "Indonesia",
    "Pakistan", "Nigeria", "Brazil", "Bangladesh",
    "Russian Federation", "Ethiopia"
]

top_countries = df[df["Country Name"].isin(countries)]
top_countries = top_countries.sort_values("2024", ascending=False)

top_countries["Population_Millions"] = top_countries["2024"] / 1_000_000

plt.bar(top_countries["Country Name"], top_countries["Population_Millions"])

plt.xlabel("Country")
plt.ylabel("Population (Millions)")
plt.title("Population of Top 10 Countries in 2024")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

