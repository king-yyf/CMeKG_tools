# coding: utf-8

l2i_dic = {"S":0, "B":1, "M":2, "E":3, "<pad>":4, "<start>":5, "<eos>":6}

i2l_dic = {0:"S", 1:"B", 2:"M", 3:"E", 4:"<pad>", 5:"<start>", 6:"<eos>"}

# 超参
max_length = 150
batch_size = 24
epochs = 10
tagset_size = len(l2i_dic)
use_cuda = False

# 路径
# train_file = './data/train.txt'
# dev_file = './data/dev.txt'
# test_file = './data/test.txt'
# medical_bert = './data/model_20'
# vocab_file = './data/model_20/vocab.txt'
# save_model_dir = './data/model/'
# medical_tool_model = './data/model/pytorch_model.pkl'  # 最终工具使用的模型
