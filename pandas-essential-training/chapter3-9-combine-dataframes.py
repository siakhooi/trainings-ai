import pandas as pd

start = pd.DataFrame({"city": ["London", "Rio", "Tokyo"],
                      "start_date": ["27th Jul, 2012", "5th Aug, 2016", "23rd July, 2021"]})
print(start)

end = pd.DataFrame({"City": ["London", "Tokyo", "Paris"],
                    "end_date": ["12th Aug, 2012", "8th Aug, 2021", "11th Aug, 2024"]})
print(end)

pd.concat([start, end], axis=0)


end = pd.DataFrame({"city": ["London", "Tokyo", "Paris"],
                    "end_date": ["12th Aug, 2012", "8th Aug, 2021", "11th Aug, 2024"]})
print(end)

df=pd.concat([start, end], axis=0)
print(df)
df=pd.concat([start, end], axis=1)
print(df)

# merge

df=pd.merge(left=start, right=end, on="city", how="inner")
print(df)

df=pd.merge(left=start, right=end, on="city", how="outer")
print(df)

df=pd.merge(left=start, right=end, on="city", how="left")
print(df)

df=pd.merge(left=start, right=end, on="city", how="right")
print(df)
