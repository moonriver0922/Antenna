# for i in range(360):
#     print(f'"a{i}",', end='')
import pandas as pd

df = pd.read_csv("output1.csv")
print(df.iloc[0:1, :])
