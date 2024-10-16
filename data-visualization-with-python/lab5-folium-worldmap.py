import folium
from folium.plugins import MarkerCluster
import pandas as pd

world_map = folium.Map()
world_map
world_map.save("worldmap.html")

canada_map = folium.Map(location=[56.130, -106.35], zoom_start=4)

canada_map
canada_map.save("canada.html")

canada_map = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles="Stamen Toner", attr="<div></div>")
canada_map.save("canada-stamen-toner.html")

canada_map = folium.Map(
    location=[56.130, -106.35], zoom_start=4, tiles="Stamen Terrain", attr="<div></div>"
)
canada_map.save("canada-stamen-terrain.html")

# Markers

canada_map = folium.Map(location=[56.1304, -106.3468], zoom_start=4)
folium.Marker(location=[51.2538, -85.3232], popup="Ontario").add_to(canada_map)
canada_map.save("canada-antario.html")

# Feature Group

canada_map = folium.Map(location=[56.130, -106.35], zoom_start=4)
ontario = folium.map.FeatureGroup()
ontario.add_child(
    folium.features.CircleMarker(
        [51.25, -85.32], radius=5, color="red", fill_color="Red"
    )
)

canada_map.add_child(ontario)
folium.Marker(location=[51.2538, -85.3232], popup="Ontario").add_to(canada_map)
canada_map.save("canada-antario-featuregroup.html")

# Multiple Markers
locations = [
    {"location": [45.4215, -75.6989], "popup": "Ottawa"},
    {"location": [53.5461, -113.4938], "popup": "Edmonton"},
    {"location": [49.2827, -123.1207], "popup": "Vancouver"},
]
for loc in locations:
    folium.Marker(location=loc["location"], popup=loc["popup"]).add_to(canada_map)
canada_map.save("canada-multiple-markers.html")

# MarkerCluster
marker_cluster = MarkerCluster().add_to(canada_map)
for loc in locations:
    folium.Marker(location=loc["location"], popup=loc["popup"]).add_to(marker_cluster)
canada_map.save("canada-marker-cluster.html")


URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Police_Department_Incidents_-_Previous_Year__2016_.csv'
df_incidents =  pd.read_csv(URL)
df_incidents.head()
df_incidents.shape
limit = 100
df_incidents = df_incidents.iloc[0:limit, :]
df_incidents.shape
# San Francisco latitude and longitude values
latitude = 37.77
longitude = -122.42
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)
sanfran_map

incidents = folium.map.FeatureGroup()
for lat, lng, in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=5,
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )
sanfran_map.add_child(incidents)
sanfran_map.save("sanfrancisco-1.html")


incidents = folium.map.FeatureGroup()
for lat, lng, in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        folium.features.CircleMarker(
            [lat, lng],
            radius=5,
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )
latitudes = list(df_incidents.Y)
longitudes = list(df_incidents.X)
labels = list(df_incidents.Category)
for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.Marker([lat, lng], popup=label).add_to(sanfran_map)
sanfran_map.add_child(incidents)
sanfran_map.save("sanfrancisco-2.html")

sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)
for lat, lng, label in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
    folium.features.CircleMarker(
        [lat, lng],
        radius=5,
        color='yellow',
        fill=True,
        popup=label,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(sanfran_map)
sanfran_map
sanfran_map.save("sanfrancisco-3.html")


sanfran_map = folium.Map(location = [latitude, longitude], zoom_start = 12)
incidents = MarkerCluster().add_to(sanfran_map)
for lat, lng, label, in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)
sanfran_map
sanfran_map.save("sanfrancisco-4.html")
