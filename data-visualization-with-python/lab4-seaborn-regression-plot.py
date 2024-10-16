import pandas as pd
import numpy as np
import seaborn as sns

df_can = pd.read_csv("resources/df_can.csv")
df_can.set_index("Country", inplace=True)

years = list(map(str, range(1980, 2014)))

df_tot = pd.DataFrame(df_can[years].sum(axis=0))

# regression plot

ax=sns.regplot(x='year', y='total', data=df_tot)
ax.figure.savefig('lab 4 seaborn regression plot 1.png')

ax=sns.regplot(x='year', y='total', data=df_tot, color='green')
ax.figure.savefig('lab 4 seaborn regression plot 2.png')

ax=sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+')
ax.figure.savefig('lab 4 seaborn regression plot 3.png')

# distribution plot

sns.countplot(x='Continent', data=df_can)

# categorical plot

sns.barplot(x='Continent', y='total', data=df_can)

# scatter plots with regression
