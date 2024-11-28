import pandas as pd

city = ["London", "Rio", "Tokyo"]
start_date = ["27th Jul, 2012", "5th Aug, 2016", "23rd July, 2021"]

# create series
pd.Series(city)

# create dataframe

df=pd.DataFrame({"City": pd.Series(city),
              "Start Date": pd.Series(start_date)})
print(df)
df=pd.DataFrame({"City": city,
              "Start Date": start_date})
print(df)

df=pd.DataFrame(zip(city, start_date))
print(df)
df=pd.DataFrame(zip(city, start_date), columns=["City", "Start Date"])
print(df)
