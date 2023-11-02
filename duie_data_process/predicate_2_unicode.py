# import json
#
#
# def chinese_to_unicode(text):
#     """将中文字符转换为Unicode表示"""
#     return ''.join(['\\u{:04x}'.format(ord(c)) for c in text])
#
#
# # 源文件路径
# source_file = 'C:\\Users\\Aki\\source\\python\\CMeKG_tools\\duie_data_process\\predicate.json'
# # 目标文件路径
# target_file = 'C:\\Users\\Aki\\source\\python\\CMeKG_tools\\duie_data_process\\predicate_unicode.json'
#
# # 读取源文件内容
# with open(source_file, 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# # 将key转换为Unicode表示
# new_data = {chinese_to_unicode(key): value for key, value in data.items()}
#
# # 保存到目标文件中
# with open(target_file, 'w', encoding='utf-8') as f:
#     json.dump(new_data, f, ensure_ascii=False, indent=2)
#
# print(f"Conversion complete. Data saved to {target_file}")

import json

# 读取文件内容
with open("C:\\Users\\Aki\\source\\python\\CMeKG_tools\\duie_data_process\\predicate.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 使用dict comprehension将每个键转换为它的Unicode表示形式
unicode_data = {key.encode('unicode_escape').decode(): value for key, value in data.items()}

# 保存转换后的数据
with open("C:\\Users\\Aki\\source\\python\\CMeKG_tools\\duie_data_process\\predicate_unicode.json", "w", encoding="utf-8") as f:
    json.dump(unicode_data, f, ensure_ascii=False, indent=2)
