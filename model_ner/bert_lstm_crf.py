# -*- coding: utf-8 -*-
# @Author: ding zeyuan
# @Date:   2019-7-5 18:24:32
from itertools import zip_longest
import torch.nn as nn
from transformers import BertModel
from model_ner.crf import CRF
from torch.autograd import Variable
import torch

class BERT_LSTM_CRF(nn.Module):
    def __init__(self, bert_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1, use_cuda):
        super(BERT_LSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeds = BertModel.from_pretrained(bert_config)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=rnn_layers, bidirectional=True, dropout=dropout_ratio, batch_first=True)
        self.rnn_layers = rnn_layers
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)
        self.embed_drop = nn.Dropout(0.5)

        self.lin = nn.Linear(2 * hidden_dim, hidden_dim)
        self.liner = nn.Linear(hidden_dim, tagset_size+2)
        self.tagset_size = tagset_size
        self.use_cuda =  use_cuda

    def rand_init_hidden(self, batch_size):
        if self.use_cuda:

            return Variable(
                torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)).cuda(), Variable(
                torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(
                torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)), Variable(
                torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim))

    def get_output_score(self, sentence, attention_mask=None):
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)
        embeds, _ = self.word_embeds(sentence, attention_mask=attention_mask)
        embed = self.embed_drop(embeds)

        hidden = self.rand_init_hidden(batch_size)
        # if embeds.is_cuda:
        #     hidden = (i.cuda() for i in hidden)
        lstm_out, hidden = self.lstm(embed, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)
        d_lstm_out = self.dropout1(lstm_out)
        lin_out = self.lin(d_lstm_out)
        l_lstm_out = self.dropout1(lin_out)
        l_out = self.liner(l_lstm_out)
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
        return lstm_feats

    def forward(self, sentence, masks):
        lstm_feats = self.get_output_score(sentence)
        scores, tag_seq = self.crf._viterbi_decode(lstm_feats, masks.byte())
        return tag_seq

    def neg_log_likelihood_loss(self, sentence, mask, tags):
        lstm_feats = self.get_output_score(sentence)
        loss_value = self.crf.neg_log_likelihood_loss(lstm_feats, mask, tags)
        batch_size = lstm_feats.size(0)
        loss_value /= float(batch_size)
        return loss_value




    def test(self, crf_scores, lengths, tag2id):
        """使用维特比算法进行解码"""
        start_id = tag2id['<start>']
        end_id = tag2id['<eos>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        # crf_scores,_ = self.forward(test_sents_tensor, lengths)
        # B:batch_size, L:max_len, T:target set size
        B, L, T = 16,450,16
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id)
        lengths = torch.LongTensor(lengths)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                        :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor([end_id] * (batch_size_t - prev_batch_size_t))
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids



