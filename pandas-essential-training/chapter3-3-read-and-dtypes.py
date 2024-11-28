import pandas as pd

filename = "olympics_1896_2004.csv"

oo = pd.read_csv(filename, skiprows=5)


ordered_medals = pd.api.types.CategoricalDtype(categories=["Bronze", "Silver", "Gold"], ordered=True)
print(ordered_medals)

dtype_mapper = {"Year": "int64",
                "City": "string",
                "Sport": "string",
                "Discipline": "string",
                "Athlete Name": "string",
                "NOC": "string",
                "Gender": "category",
                "Event": "string",
                "Event_gender": "category",
                "Medal": ordered_medals}

oo = pd.read_csv(filename, skiprows=5, dtype=dtype_mapper)
print(oo.sample(3))

print(oo.dtypes)

oo = pd.read_csv(filename, skiprows=5, dtype=dtype_mapper)
oo['Event Gender'] = oo['Event Gender'].astype("category")
print(oo.dtypes)
