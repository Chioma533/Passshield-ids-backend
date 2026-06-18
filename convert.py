import pandas as pd

df = pd.read_csv("KDDTest+.txt", header=None)
df.to_csv("KDDTest+.csv", index=False, header=False)

print("Conversion completed: KDDTest+.csv")