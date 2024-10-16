import pandas as pd

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx"
df_can = pd.read_excel(
    URL, sheet_name="Canada by Citizenship", skiprows=range(20), skipfooter=2
)

# housekeep

df_can.drop(["AREA", "REG", "DEV", "Type", "Coverage"], axis=1, inplace=True)

df_can.rename(
    columns={"OdName": "Country", "AreaName": "Continent", "RegName": "Region"},
    inplace=True,
)

df_can["Total"] = df_can.sum(axis=1, numeric_only=True)
df_can.to_csv("resources/df_can.csv")
