import json

path = "C:\\Users\\Aki\\source\\python\\CMeKG_tools\\duie_data_process\\duie_spo_train.json"

if __name__ == '__main__':
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
        print(data[0])