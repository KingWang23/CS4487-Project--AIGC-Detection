import pandas as pd

file1 = ""    # 改成你的第一个文件
file2 = ""    # 改成你的第二个文件

a = pd.read_csv(file1, header=None, names=['id', 'label'])
b = pd.read_csv(file2, header=None, names=['id', 'label'])

diff = a.merge(b, on='id', how='inner', suffixes=('_1', '_2'))
diff = diff[diff.label_1 != diff.label_2]

print(f"总共发现 {len(diff)} 个ID的label不一样！\n")







