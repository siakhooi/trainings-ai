import pandas as pd
import matplotlib.pyplot as plt

df_can = pd.read_csv("resources/df_can.csv")
df_can.set_index("Country", inplace=True)

years = list(map(str, range(1980, 2014)))

# Pie Charts

df_continents = df_can.groupby('Continent', axis=0).sum()
df_continents.head()
df_continents['Total'].plot(kind='pie',
                            figsize=(5, 6),
                            autopct='%1.1f%%',
                            startangle=90,
                            shadow=True,
                            )
plt.title('Immigration to Canada by Continent [1980 - 2013]')
plt.axis('equal')
plt.show()
plt.savefig("figure lab 3 pie chart.png")

colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1]
df_continents['Total'].plot(kind='pie',
                            figsize=(15, 6),
                            autopct='%1.1f%%',
                            startangle=90,
                            shadow=True,
                            labels=None,
                            pctdistance=1.12,
                            colors=colors_list,
                            explode=explode_list
                            )
plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12)
plt.axis('equal')
plt.legend(labels=df_continents.index, loc='upper left')
plt.show()
plt.savefig("figure lab 3 pie chart - explode property.png")

explode_list = [0.0, 0, 0, 0.1, 0.1, 0.2]
df_continents['2013'].plot(kind='pie',
                            figsize=(15, 6),
                            autopct='%1.1f%%',
                            startangle=90,
                            shadow=True,
                            labels=None,
                            pctdistance=1.12,
                            explode=explode_list
                            )
plt.title('Immigration to Canada by Continent in 2013', y=1.12)
plt.axis('equal')
plt.legend(labels=df_continents.index, loc='upper left')
plt.show()
plt.savefig("figure lab 3 pie chart - flaws.png")
