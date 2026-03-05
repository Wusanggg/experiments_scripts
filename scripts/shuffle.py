import pandas as pd

# 1. 读取CSV文件
df = pd.read_csv(r"D:\Desktop\audio\zhenyu\original_dataset\sound_with_background\csv\airport_results.csv")

# 2. 打乱数据框（DataFrame）中所有行的顺序
# `frac=1` 表示抽取100%的行，即全部行，从而达到打乱顺序的目的
# `random_state` 参数可以设置一个固定值，以便结果可复现
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. 将打乱后的数据保存到新的CSV文件
df_shuffled.to_csv(r"D:\Desktop\audio\zhenyu\original_dataset\sound_with_background\csv\shuffled_airport.csv", index=False)
