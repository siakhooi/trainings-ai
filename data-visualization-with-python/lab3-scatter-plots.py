import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_can = pd.read_csv("resources/df_can.csv")
df_can.set_index("Country", inplace=True)

years = list(map(str, range(1980, 2014)))

# Scatter Plots

df_tot = pd.DataFrame(df_can[years].sum(axis=0))
df_tot.index = map(int, df_tot.index)
df_tot.reset_index(inplace = True)
df_tot.columns = ['year', 'total']

df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')
plt.savefig("figure lab 3 scatter plot - 0.png")

plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.show()
plt.savefig("figure lab 3 scatter plot - 1.png")

x = df_tot['year']
y = df_tot['total']
fit = np.polyfit(x, y, deg=1)
fit
df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')
plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.plot(x, fit[0] * x + fit[1], color='red')
plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))
plt.show()
plt.savefig("figure lab 3 scatter plot - 2.png")

'No. Immigrants = {0:.0f} * Year + {1:.0f}'.format(fit[0], fit[1])
df_countries = df_can.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()
df_total = pd.DataFrame(df_countries.sum(axis=1))
df_total.reset_index(inplace=True)
df_total.columns = ['year', 'total']
df_total['year'] = df_total['year'].astype(int)
df_total.head()
df_total.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')
plt.savefig("figure lab 3 scatter plot - 2a.png")

plt.title('Immigration from Denmark, Norway, and Sweden to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
plt.show()
plt.savefig("figure lab 3 scatter plot - 3.png")
