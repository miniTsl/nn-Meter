import re

# 打开 txt 文件读取内容
with open('tmp.txt', 'r') as file:
    lines = file.readlines()

# 定义匹配 "All + 数字 + models" 开头的正则表达式
pattern = re.compile(r'^\(nn-Meter\) All \d+ models')

# 结果列表，用于存储匹配到的行和其后的 6 行
results = []

# 遍历所有行，找到符合条件的行及其后的 6 行
for i, line in enumerate(lines):
    if pattern.match(line):
        results.extend(lines[i:i+7])
        results.extend(['\n'])

# 将结果写入到一个新的 txt 文件中
with open('./filtered_output.txt', 'w') as output_file:
    output_file.writelines(results)

print("提取完成，结果已保存到 'filtered_output.txt'")