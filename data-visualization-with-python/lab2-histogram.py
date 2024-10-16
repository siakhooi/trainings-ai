import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Histogram
df_can = pd.read_csv("resources/df_can.csv")
df_can.set_index("Country", inplace=True)
years = list(map(str, range(1980, 2014)))

count, bin_edges = np.histogram(df_can["2013"])
df_can["2013"].plot(kind="hist", figsize=(8, 5))
plt.title("Histogram of Immigration from 195 Countries in 2013")
plt.ylabel("Number of Countries")
plt.xlabel("Number of Immigrants")
plt.show()
plt.savefig("figure lab 2 histogram.png")

count, bin_edges = np.histogram(df_can["2013"])
df_can["2013"].plot(kind="hist", figsize=(8, 5), xticks=bin_edges)
plt.title("Histogram of Immigration from 195 countries in 2013")
plt.ylabel("Number of Countries")
plt.xlabel("Number of Immigrants")
plt.show()
plt.savefig("figure lab 2 histogram bin edge.png")

df_t = df_can.loc[["Denmark", "Norway", "Sweden"], years].transpose()
df_t.head()
df_t.plot(kind="hist", figsize=(10, 6))
plt.title("Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013")
plt.ylabel("Number of Years")
plt.xlabel("Number of Immigrants")
plt.show()
plt.savefig("figure lab 2 histogram denmark, norway, sweden.png")

count, bin_edges = np.histogram(df_t, 15)
df_t.plot(
    kind="hist",
    figsize=(10, 6),
    bins=15,
    alpha=0.6,
    xticks=bin_edges,
    color=["coral", "darkslateblue", "mediumseagreen"],
)
plt.title("Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013")
plt.ylabel("Number of Years")
plt.xlabel("Number of Immigrants")
plt.show()
plt.savefig("figure lab 2 histogram denmark, norway, sweden 2.png")

# stacked Histogram

count, bin_edges = np.histogram(df_t, 15)
xmin = bin_edges[0] - 10
xmax = bin_edges[-1] + 10
df_t.plot(
    kind="hist",
    figsize=(10, 6),
    bins=15,
    xticks=bin_edges,
    color=["coral", "darkslateblue", "mediumseagreen"],
    stacked=True,
    xlim=(xmin, xmax),
)
plt.title("Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013")
plt.ylabel("Number of Years")
plt.xlabel("Number of Immigrants")
plt.show()
plt.savefig("figure lab 2 histogram denmark, norway, sweden 3.png")
