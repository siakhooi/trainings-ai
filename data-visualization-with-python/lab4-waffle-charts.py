import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pywaffle import Waffle
import matplotlib.patches as mpatches

df_can = pd.read_csv("resources/df_can.csv")
df_can.set_index("Country", inplace=True)

years = list(map(str, range(1980, 2014)))

# Waffle Charts

df_dsn = df_can.loc[['Denmark', 'Norway', 'Sweden'], :]
df_dsn
total_values = df_dsn['Total'].sum()
category_proportions = df_dsn['Total'] / total_values
pd.DataFrame({"Category Proportion": category_proportions})

width = 40
height = 10
total_num_tiles = width * height
print(f'Total number of tiles is {total_num_tiles}.')
tiles_per_category = (category_proportions * total_num_tiles).round().astype(int)
pd.DataFrame({"Number of tiles": tiles_per_category})

waffle_chart = np.zeros((height, width), dtype = np.uint)
category_index = 0
tile_index = 0

for col in range(width):
    for row in range(height):
        tile_index += 1
        if tile_index > sum(tiles_per_category[0:category_index]):
            category_index += 1
        waffle_chart[row, col] = category_index
print ('Waffle chart populated!')

waffle_chart

fig = plt.figure()
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()
plt.show()
fig = plt.figure()
plt.savefig("figure lab 4 waffle chart - 1.png")

colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()
ax = plt.gca()

# set minor ticks
ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

# add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])
plt.show()
plt.savefig("figure lab 4 waffle chart - 2.png")

# instantiate a new figure object
fig = plt.figure()

# use matshow to display the waffle chart
colormap = plt.cm.coolwarm
plt.matshow(waffle_chart, cmap=colormap)
plt.colorbar()

ax = plt.gca()

ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

# add gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

plt.xticks([])
plt.yticks([])

values_cumsum = np.cumsum(df_dsn['Total'])
total_values = values_cumsum[len(values_cumsum) - 1]

legend_handles = []
for i, category in enumerate(df_dsn.index.values):
    label_str = category + ' (' + str(df_dsn['Total'][i]) + ')'
    color_val = colormap(float(values_cumsum[i])/total_values)
    legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

plt.legend(handles=legend_handles,
           loc='lower center',
           ncol=len(df_dsn.index.values),
           bbox_to_anchor=(0., -0.2, 0.95, .1)
          )
plt.show()
plt.savefig("figure lab 4 waffle chart - 4.png")
