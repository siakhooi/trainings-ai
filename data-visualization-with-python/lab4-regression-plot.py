import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_can = pd.read_csv("resources/df_can.csv")
df_can.set_index("Country", inplace=True)

years = list(map(str, range(1980, 2014)))

# Regression Plot

df_tot = pd.DataFrame(df_can[years].sum(axis=0))
df_tot.index = map(float, df_tot.index)
df_tot.reset_index(inplace=True)
df_tot.columns = ['year', 'total']
df_tot.head()

sns.regplot(x='year', y='total', data=df_tot)

sns.regplot(x='year', y='total', data=df_tot, color='green')
plt.show()
plt.savefig("figure lab 4 regression plot - 1.png")

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.show()
plt.savefig("figure lab 4 regression plot - 2.png")

plt.figure(figsize=(15, 10))
sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
plt.show()
plt.savefig("figure lab 4 regression plot - 3.png")

plt.figure(figsize=(15, 10))
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})

ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()
plt.savefig("figure lab 4 regression plot - 4.png")

plt.figure(figsize=(15, 10))
sns.set(font_scale=1.5)

ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()
plt.savefig("figure lab 4 regression plot - 5.png")

plt.figure(figsize=(15, 10))
sns.set(font_scale=1.5)
sns.set_style('ticks')
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()
plt.savefig("figure lab 4 regression plot - 6.png")

plt.figure(figsize=(15, 10))
sns.set(font_scale=1.5)
sns.set_style('whitegrid')
ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
ax.set(xlabel='Year', ylabel='Total Immigration')
ax.set_title('Total Immigration to Canada from 1980 - 2013')
plt.show()
plt.savefig("figure lab 4 regression plot - 7.png")
