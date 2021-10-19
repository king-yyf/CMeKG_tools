# -*- coding: utf-8 -*-

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from utils import load_vocab, load_data, recover_label, get_ner_fmeasure, save_model, load_model
from ner_constant import *
from model_ner import BERT_LSTM_CRF


print('device',device)
# if torch.cuda.is_available():
#     device = torch.device("cuda", 2)
#     print('device',device)
#     use_cuda = True
# else:
#     device = torch.device("cpu")
#     use_cuda = False
vocab = load_vocab(vocab_file)
vocab_reverse = {v:k for k, v in vocab.items()}

print('max_length',max_length)


train_data = load_data(train_file, max_length=max_length, label_dic=l2i_dic, vocab=vocab)
train_ids = torch.LongTensor([temp.input_id for temp in train_data[1500:]])
train_masks = torch.LongTensor([temp.input_mask for temp in train_data[1500:]])
train_tags = torch.LongTensor([temp.label_id for temp in train_data[1500:]])
train_lenghts = torch.LongTensor([temp.lenght for temp in train_data[1500:]])
train_dataset = TensorDataset(train_ids, train_masks, train_tags,train_lenghts)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

dev_data = load_data(dev_file, max_length=max_length, label_dic=l2i_dic, vocab=vocab)
dev_ids = torch.LongTensor([temp.input_id for temp in dev_data[:1500]])
dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data[:1500]])
dev_tags = torch.LongTensor([temp.label_id for temp in dev_data[:1500]])
dev_lenghts = torch.LongTensor([temp.lenght for temp in dev_data[:1500]])
dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags,dev_lenghts)
dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=batch_size)


test_data = load_data(test_file, max_length=max_length, label_dic=l2i_dic, vocab=vocab)
test_ids = torch.LongTensor([temp.input_id for temp in test_data])
test_masks = torch.LongTensor([temp.input_mask for temp in test_data])
test_tags = torch.LongTensor([temp.label_id for temp in test_data])
test_lenghts = torch.LongTensor([temp.lenght for temp in test_data])


test_dataset = TensorDataset(test_ids, test_masks, test_tags,test_lenghts)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)




######测试函数
def evaluate(medel, dev_loader):
    medel.eval()
    pred = []
    gold = []
    print('evaluate')
    with torch.no_grad():
        for i, dev_batch in enumerate(dev_loader):
            sentence, masks, tags , lengths = dev_batch
            sentence, masks, tags, lengths = Variable(sentence), Variable(masks), Variable(tags), Variable(lengths)
            if use_cuda:
                sentence = sentence.to(device)
                masks = masks.to(device)
                tags = tags.to(device)

            predict_tags = medel(sentence, masks)
            loss = model.neg_log_likelihood_loss(sentence, masks, tags)

            pred.extend([t for t in predict_tags.tolist()])
            gold.extend([t for t in tags.tolist()])

        pred_label,gold_label = recover_label(pred, gold, l2i_dic,i2l_dic)
        print('dev loss {}'.format(loss.item()))
        pred_label_1 = [t[1:] for t in pred_label]
        gold_label_1 = [t[1:] for t in gold_label]
        acc,p, r, f = get_ner_fmeasure(gold_label_1,pred_label_1)
        print('p: {}，r: {}, f: {}'.format(p, r, f))
        return p, r, f

# test 函数
def evaluate_test(medel,test_loader,dev_f):
    medel.eval()
    pred = []
    gold = []
    print('test')
    with torch.no_grad():
        for i, dev_batch in enumerate(test_loader):
            sentence, masks, tags, lengths = dev_batch
            sentence, masks, tags , lengths = Variable(sentence), Variable(masks), Variable(tags),Variable(lengths)
            if use_cuda:
                sentence = sentence.to(device)
                masks = masks.to(device)
                tags = tags.to(device)
            predict_tags = medel(sentence, masks)

            pred.extend([t for t in predict_tags.tolist()])
            gold.extend([t for t in tags.tolist()])

        pred_label, gold_label = recover_label(pred, gold, l2i_dic,i2l_dic)
        pred_label_2 = [t[1:] for t in pred_label]
        gold_label_2 = [t[1:] for t in gold_label]
        fw = open('data/predict_result'+str(float('%.3f'%dev_f))+'bert.txt','w')
        for i in pred_label_2:
            for j in range(len(i)-1):
                fw.write(i[j])
                fw.write(' ')
            fw.write(i[len(i)-1])
            fw.write('\n')
        acc,p, r, f = get_ner_fmeasure(gold_label_2,pred_label_2)
        print('p: {}，r: {}, f: {}'.format(p, r, f))
    return p, r, f



model = BERT_LSTM_CRF('./data/my_bert', tagset_size, 768, 200, 2,
                      dropout_ratio=0.5, dropout1=0.5, use_cuda = use_cuda)

if use_cuda:
    model.to(device)

optimizer = getattr(optim, 'Adam')
optimizer = optimizer(model.parameters(), lr=0.000005, weight_decay=0.00005)

best_f = -100
model_name = save_model_dir + '0518' + str(float('%.3f' % best_f)) + ".pkl"
print(model_name)

for epoch in range(epochs):
    print('epoch: {}，train'.format(epoch))
    for i, train_batch in enumerate(tqdm(train_loader)):
        sentence, masks, tags , lengths= train_batch

        sentence, masks, tags , lengths = Variable(sentence), Variable(masks), Variable(tags), Variable(lengths)

        if use_cuda:
            sentence = sentence.to(device)
            masks = masks.to(device)
            tags = tags.to(device)
        model.train()
        optimizer.zero_grad()
        loss = model.neg_log_likelihood_loss(sentence, masks, tags)
        loss.backward()
        optimizer.step()

    print('epoch: {}，train loss: {}'.format(epoch, loss.item()))
    p, r, f = evaluate(model,dev_loader)


    if f > best_f:
        best_f = f
        _, _, _ = evaluate_test(model, test_loader, loss.item())
        model_name = save_model_dir + 'new' + str(float('%.3f' % best_f)) + ".pkl"
        torch.save(model.state_dict(), model_name)











