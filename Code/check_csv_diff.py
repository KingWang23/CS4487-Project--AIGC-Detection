import pandas as pd

# ================== 请修改这里 ==================
csv_file1 = ''   # 第一个 csv 文件路径
csv_file2 = ''   # 第二个 csv 文件路径
# ===============================================

# 读取两个文件（假设第一列是 ID，第二列是 label）
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

# 把 ID 设置为索引，便于比较
df1 = df1.set_index('ID')
df2 = df2.set_index('ID')

# 只保留两个文件都有的 ID（交集）
common_ids = df1.index.intersection(df2.index)

print(f"两个文件共有 {len(common_ids)} 个相同的 ID\n")

# 找出 label 不一致的 ID
diff = df1.loc[common_ids, 'label'] != df2.loc[common_ids, 'label']
diff_ids = diff[diff].index

if len(diff_ids) == 0:
    print("✔ 所有相同 ID 的 label 都一致！")
else:
    print(f"✖ 发现 {len(diff_ids)} 个 ID 的 label 不一致：\n")
    print("-" * 60)
    print(f"{'ID':<20} {'文件1 label':<12} {'文件2 label':<12}")
    print("-" * 60)
    for id_ in diff_ids:
        label1 = df1.loc[id_, 'label']
        label2 = df2.loc[id_, 'label']
        print(f"{id_:<20} {str(label1):<12} {str(label2):<12}")

    # 可选：把差异保存到新 csv
    result = pd.DataFrame({
        'ID': diff_ids,
        'label_file1': df1.loc[diff_ids, 'label'].values,
        'label_file2': df2.loc[diff_ids, 'label'].values
    })
    result.to_csv('label_differences.csv', index=False)
    print("\n差异已保存到 label_differences.csv")