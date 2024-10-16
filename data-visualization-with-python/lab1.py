import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

print(plt.style.available)
mpl.style.use(["ggplot"])

df_can = pd.read_csv("resources/df_can.csv")

df_can.set_index("Country", inplace=True)
df_can.index.name = None
df_can.columns = list(map(str, df_can.columns))

# Line Pots (Series/Dataframe)

years = list(map(str, range(1980, 2014)))
haiti = df_can.loc["Haiti", years]
f = haiti.plot()
f.get_figure().savefig("figure lab 1 haiti.png")

haiti.index = haiti.index.map(int)
haiti.plot(kind="line")
plt.title("Immigration from Haiti")
plt.ylabel("Number of immigrants")
plt.xlabel("Years")
plt.show()
plt.savefig("figure lab 1 haiti line labels.png")

haiti.plot(kind="line")
plt.title("Immigration from Haiti")
plt.ylabel("Number of Immigrants")
plt.xlabel("Years")
plt.text(2000, 6000, "2010 Earthquake")
plt.show()
plt.savefig("figure lab 1 2010 Earthquake.png")
