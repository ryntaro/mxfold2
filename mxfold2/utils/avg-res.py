import pandas as pd
import sys

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# 1行目にヘッダーがある前提で読み込む
df = pd.read_csv(sys.argv[1])

# すべての数値列に対して統計量を出す
print(df.describe())
