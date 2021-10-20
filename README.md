# CMeKG 工具 代码及模型


Index
---
<!-- TOC -->

- [CMeKG工具](#cmekg工具)
  - [模型下载](#模型下载)
- [依赖库](#依赖库)
- [模型使用](#模型使用)
  - [关系抽取](#医学关系抽取)
  - [医学实体识别](#医学实体识别)
  - [医学文本分词](#医学文本分词)


<!-- /TOC -->


## cmekg工具

[CMeKG网站](https://cmekg.pcl.ac.cn/)

中文医学知识图谱CMeKG
CMeKG（Chinese Medical Knowledge Graph）是利用自然语言处理与文本挖掘技术，基于大规模医学文本数据，以人机结合的方式研发的中文医学知识图谱。

CMeKG 中主要模型工具包括 医学文本分词，医学实体识别和医学关系抽取。这里是三种工具的代码、模型和使用方法。

### 模型下载

由于依赖和训练好的的模型较大，将模型放到了百度网盘中，链接如下，按需下载。

RE：链接:https://pan.baidu.com/s/1cIse6JO2H78heXu7DNewmg  密码:4s6k


NER: 链接:https://pan.baidu.com/s/16TPSMtHean3u9dJSXF9mTw  密码:shwh


分词：链接:https://pan.baidu.com/s/1bU3QoaGs2IxI34WBx7ibMQ  密码:yhek

## 依赖库

- json
- random
- numpy
- torch
- transformers
- gc
- re
- time
- tqdm

## 模型使用

### 医学关系抽取

**依赖文件**

-  pytorch_model.bin : 医学文本预训练的 BERT-base model
-  vocab.txt
-  config.json
-  model_re.pkl: 训练好的关系抽取模型文件，包含了模型参数、优化器参数等
-  predicate.json 

**使用方法**

配置参数在medical_re.py的class config里，首先在medical_re.py的class config里修改各个文件路径

- 训练

```python
import medical_re
medical_re.load_schema()
medical_re.run_train()
```

model_re/train_example.json 是训练文件示例

- 使用

```python
import medical_re
medical_re.load_schema()
model4s, model4po = medical_re.load_model()

text = '据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人。'  # content是输入的一段文字
res = medical_re.get_triples(text, model4s, model4po)
print(json.dumps(res, ensure_ascii=False, indent=True))
```

- 执行结果

```
[
 {
  "text": "据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、=乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人",
  "triples": [
   [
    "新冠肺炎",
    "临床表现",
    "肺炎"
   ],
   [
    "新冠肺炎",
    "临床表现",
    "发热"
   ],
   [
    "新冠肺炎",
    "临床表现",
    "咳嗽"
   ],
   [
    "新冠肺炎",
    "临床表现",
    "胸闷"
   ],
   [
    "新冠肺炎",
    "临床表现",
    "乏力"
   ],
   [
    "新冠肺炎",
    "病因",
    "自身免疫系统缺陷"
   ],
   [
    "新冠肺炎",
    "病因",
    "人传人"
   ]
  ]
 }
]
```

### 医学实体识别

调整的参数和模型在ner_constant.py中

**训练**

python3 train_ner.py


**使用示例**


medical_ner 类提供两个接口测试函数

- predict_sentence(sentence): 测试单个句子，返回:{"实体类别"：“实体”},不同实体以逗号隔开
- predict_file(input_file, output_file): 测试整个文件
文件格式每行待提取实体的句子和提取出的实体{"实体类别"：“实体”},不同实体以逗号隔开

```python
from run import medical_ner

#使用工具运行
my_pred=medical_ner()
#根据提示输入单句：“高血压病人不可食用阿莫西林等药物”
sentence=input("输入需要测试的句子:")
my_pred.predict_sentence("".join(sentence.split()))

#输入文件(测试文件，输出文件)
my_pred.predict_file("my_test.txt","outt.txt")
```

### 医学文本分词

调整的参数和模型在cws_constant.py中

**训练**

python3 train_cws.py


**使用示例**


medical_cws 类提供两个接口测试函数

- predict_sentence(sentence): 测试单个句子，返回:{"实体类别"：“实体”},不同实体以逗号隔开
- predict_file(input_file, output_file): 测试整个文件
文件格式每行待提取实体的句子和提取出的实体{"实体类别"：“实体”},不同实体以逗号隔开

```python
from run import medical_cws

#使用工具运行
my_pred=medical_cws()
#根据提示输入单句：“高血压病人不可食用阿莫西林等药物”
sentence=input("输入需要测试的句子:")
my_pred.predict_sentence("".join(sentence.split()))

#输入文件(测试文件，输出文件)
my_pred.predict_file("my_test.txt","outt.txt")
```


