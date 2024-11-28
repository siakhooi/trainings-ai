import pandas as pd

city = ["London", "Rio", "Tokyo"]
start_date = ["07-27-2012", "5th Aug, 2016", "23rd Jul, 2021"]
end_date = ["12th Aug, 2012", "21-08-2016", "8th Aug, 2021"]

games = pd.DataFrame(zip(city, start_date, end_date), columns=["City", "Start Date", "End Date"])
print(games)
print(games.dtypes)


pd.to_datetime(games["Start Date"], format='mixed')

games["Start Date"] = pd.to_datetime(games["Start Date"], format='mixed')
games["End Date"] = pd.to_datetime(games["End Date"], format='mixed')
games["City"] = games.City.astype("string")
print(games)
print(games.dtypes)

x=games["End Date"] - games["Start Date"]
print(x)

games = games.assign(duration=games["End Date"] - games["Start Date"])
print(games)
print(games.dtypes)
