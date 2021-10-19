# -*- coding: utf-8 -*-
import torch
# tag-entity:{d:疾病 s:临床表现 b:身体 e:医疗设备 p:医疗程序 m:微生物类 k:科室 i:医学检验项目 y:药物}
l2i_dic = {"o": 0, "d-B": 1, "d-M": 2, "d-E": 3, "s-B": 4, "s-M": 5,"s-E": 6,
           "b-B": 7, "b-M": 8, "b-E": 9, "e-B": 10, "e-M": 11, "e-E": 12, "p-B": 13, "p-M": 14, "p-E": 15, "m-B": 16,"m-M": 17,
           "m-E": 18, "k-B": 19, "k-M": 20, "k-E": 21, "i-B": 22, "i-M": 23,"i-E": 24, "y-B": 25, "y-M": 26, "y-E": 27,"<pad>":28,"<start>": 29, "<eos>": 30}

i2l_dic = {0: "o", 1: "d-B", 2: "d-M", 3: "d-E", 4: "s-B", 5: "s-M",
           6: "s-E", 7: "b-B", 8: "b-M", 9: "b-E", 10: "e-B", 11: "e-M", 12: "e-E",13:"p-B", 14:"p-M", 15:"p-E",
           16: "m-B", 17: "m-M", 18: "m-E", 19: "k-B",20: "k-M", 21: "k-E",
          22: "i-B", 23: "i-M", 24: "i-E", 25: "y-B", 26: "y-M", 27: "y-E", 28: "<pad>",29:"<start>", 30:"<eos>"}


# train_file = './data/train_data.txt'
# dev_file = './data/val_data.txt'
# test_file = './data/test_data.txt'
# vocab_file = './data/my_bert/vocab.txt'

# save_model_dir =  './data/model/'
# medical_tool_model = './data/model/model.pkl'
max_length = 450
batch_size = 1
epochs = 30
tagset_size = len(l2i_dic)
use_cuda = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")