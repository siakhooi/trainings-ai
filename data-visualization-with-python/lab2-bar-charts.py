import pandas as pd
import matplotlib.pyplot as plt

df_can = pd.read_csv("resources/df_can.csv")
df_can.set_index("Country", inplace=True)

years = list(map(str, range(1980, 2014)))

# Bar Charts

df_iceland = df_can.loc["Iceland", years]
df_iceland.head()

df_iceland.plot(kind="bar", figsize=(10, 6))
plt.xlabel("Year")
plt.ylabel("Number of immigrants")
plt.title("Icelandic immigrants to Canada from 1980 to 2013")
plt.show()
plt.savefig(
    "figure lab 2 barchart Icelandic immigrants to Canada from 1980 to 2013 - 1.png"
)


df_iceland.plot(kind="bar", figsize=(10, 6), rot=90)
plt.xlabel("Year")
plt.ylabel("Number of Immigrants")
plt.title("Icelandic Immigrants to Canada from 1980 to 2013")
plt.annotate(
    "",
    xy=(32, 70),
    xytext=(28, 20),
    xycoords="data",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="blue", lw=2),
)
plt.show()
plt.savefig(
    "figure lab 2 barchart Icelandic immigrants to Canada from 1980 to 2013 - 2.png"
)

df_iceland.plot(kind="bar", figsize=(10, 6), rot=90)
plt.xlabel("Year")
plt.ylabel("Number of Immigrants")
plt.title("Icelandic Immigrants to Canada from 1980 to 2013")
plt.annotate(
    "",
    xy=(32, 70),
    xytext=(28, 20),
    xycoords="data",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="blue", lw=2),
)
plt.annotate(
    "2008 - 2011 Financial Crisis",
    xy=(28, 30),
    rotation=72.5,
    va="bottom",
    ha="left",
)
plt.show()
plt.savefig(
    "figure lab 2 barchart Icelandic immigrants to Canada from 1980 to 2013 - 3.png"
)
