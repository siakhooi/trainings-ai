import pandas as pd
import matplotlib.pyplot as plt

df_top5 = pd.read_csv("resources/df_top5.csv")
df_top5.index = df_top5.index.map(int)
years = list(map(str, range(1980, 2014)))

# Area Plots

df_top5.plot(kind="area", stacked=False, figsize=(20, 10))
plt.title("Immigration Trend of Top 5 Countries")
plt.ylabel("Number of Immigrants")
plt.xlabel("Years")
plt.show()
plt.savefig("figure lab 2 area plot top 5.png")

df_top5.plot(kind="area", alpha=0.25, stacked=False, figsize=(20, 10))
plt.title("Immigration Trend of Top 5 Countries")
plt.ylabel("Number of Immigrants")
plt.xlabel("Years")
plt.show()
plt.savefig("figure lab 2 area plot top 5 half transparent.png")
