import pandas as pd

# 读取文件
df_gt = pd.read_csv(r"D:\Desktop\audio\zhenyu\original_dataset\sound_with_background\csv\shuffled_residential_street_gt.csv")
df_labeled = pd.read_csv(r"D:\Desktop\audio\zhenyu\original_dataset\sound_with_background\shuffled_residential_street_labeled_qwen.csv")

# 以文件名、时间戳和文本内容作为唯一标识进行合并，确保行对应
merged = pd.merge(
    df_gt, 
    df_labeled, 
    on=['Filename', 'Timestamp', 'Transcription'], 
    suffixes=('_gt', '_labeled')
)

# 筛选 Label 不一致的行
diff = merged[merged['Label_gt'] != merged['Label_labeled']]

if diff.empty:
    print("所有 Label 均一致。")
else:
    print(f"发现 {len(diff)} 处不同：")
    print(diff[['Timestamp', 'Transcription', 'Label_gt', 'Label_labeled']])

# 打印各文件 Label 分布情况
print("\nGT 文件 Label 计数:")
print(df_gt['Label'].value_counts())
print("\nLabeled 文件 Label 计数:")
print(df_labeled['Label'].value_counts())
