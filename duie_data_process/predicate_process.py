import json

# 源文件路径
source_file = 'C:\\Users\\Aki\\source\\python\\CMeKG_tools\\duie_data_process\\duie_schema.json'
# 目标文件路径
target_file = 'C:\\Users\\Aki\\source\\python\\CMeKG_tools\\duie_data_process\\predicate.json'

# 定义一个空字典来存储转换后的数据
result = {}

# 打开源文件进行读取
with open(source_file, 'r', encoding='utf-8') as f:
    for line in f:
        # 对每一行进行json解码
        data = json.loads(line)
        # 提取predicate字段并将其值设置为空字符串
        result[data['predicate']] = ""

# 将结果保存到目标文件中
with open(target_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Conversion complete. Data saved to {target_file}")
