# import pandas as pd
# import sys

# df = pd.read_csv(sys.argv[1], names=("name", "length", "elapsed_time", "score", "TP", "TN", "FP", "FN", "SEN", "PPV", "F", "MCC"))
# print(df[["length", "elapsed_time", "SEN", "PPV", "F"]].describe())

import pandas as pd
import sys


pd.set_option("display.max_columns", None)  # 列を省略しない
pd.set_option("display.width", None)       # 横幅も制限しない

df = pd.read_csv(sys.argv[1], names=(
    "name", "length", "elapsed_time", "score",
    "TP", "TN", "FP", "FN", "SEN", "PPV", "F", "MCC",
    "mse", "r2", "corr"  # Multitask の場合だけ出てくる
))

# 共通で欲しい列
cols = ["length", "elapsed_time", "SEN", "PPV", "F", "MCC"]

# もし SHAPE 指標が含まれていれば追加
for extra in ["mse", "r2", "corr"]:
    if extra in df.columns:
        cols.append(extra)

print(df[cols].describe())
