
import folium
import json
import urllib
import pandas as pd
import numpy as np

df_can = pd.read_csv("resources/df_can.csv")

df_can.columns = list(map(str, df_can.columns))
df_can.set_index("Country", inplace=True)
years = list(map(str, range(1980, 2014)))

df_can.head()

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/world_countries.json'
data = urllib.request.urlopen(URL)
world_geo = json.load(data)

print('GeoJSON file loaded!')

world_map = folium.Map(location=[0, 0], zoom_start=2)

world_map.choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Immigration to Canada'
)
world_map
world_map.save("lab-5-folium-choroplethmap-1.html")


threshold_scale = np.linspace(df_can['Total'].min(),
                              df_can['Total'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist()
threshold_scale[-1] = threshold_scale[-1] + 1

world_map = folium.Map(location=[0, 0], zoom_start=2)
world_map.choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    threshold_scale=threshold_scale,
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Immigration to Canada',
    reset=True
)
world_map
world_map.save("lab-5-folium-choroplethmap-2.html")
