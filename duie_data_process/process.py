import json

input_file_path = 'C:\\Users\\Aki\\source\\python\\CMeKG_tools\\duie_data_process\\duie_train.json'
output_file_path = 'C:\\Users\\Aki\\source\\python\\CMeKG_tools\\duie_data_process\\duie_spo_train.json'


def transform_data(input_path, output_path):
    # 定义一个列表来存放所有的转换后的数据
    result = []

    with open(input_path, 'r', encoding='utf-8') as f:
        # 使用enumerate函数来获取当前行号
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            text = data['text']
            spo_list = [
                [spo['subject'], spo['predicate'], spo['object']['@value']]
                for spo in data['spo_list']
            ]

            result.append({
                'text': text,
                'spo_list': spo_list
            })

            # 每处理100行，打印一次进度
            if i % 100 == 0:
                print(f"Processed {i} lines.")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print("Transformation completed.")


if __name__ == '__main__':
    transform_data(input_file_path, output_file_path)
