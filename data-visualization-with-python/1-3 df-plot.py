import pandas as pd  # primary data structure library

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/Canada.xlsx"
df_can = pd.read_excel(
    URL, sheet_name="Canada by Citizenship", skiprows=range(20), skipfooter=2
)

india_china_df=df_can

india_china_df.plot(kind="line")

india_china_df["India"].plot(kind="hist")
