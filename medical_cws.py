# coding:utf-8
import codecs
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from utils import load_vocab
from cws_constant import *

from model_cws import BERT_LSTM_CRF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class medical_seg(object):
    def __init__(self):
        self.NEWPATH = '/Users/yangyf/workplace/model/medical_cws/pytorch_model.pkl'
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
            self.use_cuda = True
        else:
            self.device = torch.device("cpu")
            self.use_cuda = False

        self.vocab = load_vocab('/Users/yangyf/workplace/model/medical_cws/vocab.txt')
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

        self.model = BERT_LSTM_CRF('/Users/yangyf/workplace/model/medical_cws', tagset_size, 768, 200, 2,
                              dropout_ratio=0.5, dropout1=0.5, use_cuda=use_cuda)

        if use_cuda:
            self.model.cuda()

    def from_input(self, input_str):
        # 单行的输入
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        text = ['[CLS]'] + [x for x in input_str] + ['[SEP]']
        raw_text.append(text)
        cur_len = len(text)
        # raw_textid = [self.vocab[x] for x in text] + [0] * (max_length - cur_len)
        raw_textid = [self.vocab[x] for x in text if self.vocab.__contains__(x)] + [0] * (max_length - cur_len)
        textid.append(raw_textid)
        raw_textmask = [1] * cur_len + [0] * (max_length - cur_len)
        textmask.append(raw_textmask)
        textlength.append([cur_len])
        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength

    def from_txt(self, input_path):
        # 多行输入
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line) > 148:
                    line = line[:148]
                temptext = ['[CLS]'] + [x for x in line[:-1]] + ['[SEP]']
                cur_len = len(temptext)
                raw_text.append(temptext)

                tempid = [self.vocab[x] for x in temptext[:cur_len]] + [0] * (max_length - cur_len)
                textid.append(tempid)
                textmask.append([1] * cur_len + [0] * (max_length - cur_len))
                textlength.append([cur_len])

        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength

    def recover_to_text(self, pred, raw_text):
        # 输入[标签list]和[原文list],batch为1
        pred = [i2l_dic[t.item()] for t in pred[0]]
        pred = pred[:len(raw_text)]
        pred = pred[1:-1]
        raw_text = raw_text[1:-1]
        raw = ""
        res = ""
        for tag, char in zip(pred, raw_text):
            res += char
            if tag in ["S", 'E']:
                res += ' '
            raw += char
        return raw, res

    def predict_sentence(self, sentence):
        if sentence == '':
            print("输入为空！请重新输入")
            return
        if len(sentence) > 148:
            print("输入句子过长，请输入小于148的长度字符！")
            sentence = sentence[:148]
        raw_text, test_ids, test_masks, test_lengths = self.from_input(sentence)

        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        # self.model.load_state_dict(torch.load(self.NEWPATH, map_location={'cuda:0': str(self.device)}))
        self.model.load_state_dict(torch.load(self.NEWPATH,map_location=self.device))
        self.model.eval()

        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
            if use_cuda:
                sentence = sentence.cuda()
                masks = masks.cuda()

            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()

            raw, res = self.recover_to_text(predict_tags, batch_raw_text)
            #print("输入：", raw)
            #print("结果：", res)
        return res

    def predict_file(self, input_file, output_file):
        # raw_text, test_ids, test_masks, test_lengths = self.from_txt("./data/raw_text.txt")
        raw_text, test_ids, test_masks, test_lengths = self.from_txt(input_file)

        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location={'cuda:0': str(self.device)}))
        self.model.eval()

        op_file = codecs.open(output_file, 'w', 'utf-8')
        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
            if use_cuda:
                sentence = sentence.cuda()
                masks = masks.cuda()

            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()

            raw, res = self.recover_to_text(predict_tags, batch_raw_text)
            op_file.write(res + '\n')

        op_file.close()
        print('处理完成！')
        print("results have been stored in {}".format(output_file))


if __name__ == "__main__":

    meg = medical_seg()
    # meg.predict_file('./data/raw_text.txt', './data/out_raw.txt')
    res = meg.predict_sentence("肾上腺由皮质和髓质两个功能不同的内分泌器官组成，皮质分泌肾上腺皮质激素，髓质分泌儿茶酚胺激素。")
    print(res)
