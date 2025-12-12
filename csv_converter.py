import pandas as pd

read_file = pd.read_excel("data_movies_clean.xlsx")
read_file.to_csv("data_movies_clean.csv", index=None, header=True)