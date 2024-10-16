import pandas as pd

df_can = pd.read_csv("resources/df_can.csv")

df_can.columns = list(map(str, df_can.columns))
df_can.set_index("Country", inplace=True)
years = list(map(str, range(1980, 2014)))

df_can.sort_values(["Total"], ascending=False, axis=0, inplace=True)
df_top5 = df_can.head()
df_top5 = df_top5[years].transpose()

df_top5.to_csv("resources/df_top5.csv")
