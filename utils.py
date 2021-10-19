# coding: utf-8
from cws_constant import *



class InputFeatures(object):
    def __init__(self, text, label, input_id, label_id, input_mask, length):
        self.text = text
        self.label = label
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask
        self.lenght = length


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def load_file(file_path):
    contents = open(file_path, encoding='utf-8').readlines()
    text = []
    label = []
    texts = []
    labels = []
    for line in contents:
        if line != '\n':
            line = line.strip().split('\t')
            text.append(line[0])
            label.append(line[-1])
        else:
            texts.append(text)
            labels.append(label)
            text = []
            label = []
    return texts, labels


def load_data(file_path, max_length, label_dic, vocab):
    # 返回InputFeatures的list
    texts, labels = load_file(file_path)
    assert len(texts) == len(labels)
    result = []
    for i in range(len(texts)):
        assert len(texts[i]) == len(labels[i])
        token = texts[i]
        label = labels[i]
        if len(token) > max_length - 2:
            token = token[0:(max_length - 2)]
            label = label[0:(max_length - 2)]

        tokens_f = ['[CLS]'] + token + ['[SEP]']
        label_f = ["<start>"] + label + ['<eos>']

        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        label_ids = [label_dic[i] for i in label_f]
        input_mask = [1] * len(input_ids)

        length = [len(tokens_f)]

        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_dic['<pad>'])
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(label_ids) == max_length
        feature = InputFeatures(text=tokens_f, label=label_f, input_id=input_ids, input_mask=input_mask,
                                label_id=label_ids, length=length)
        result.append(feature)

    return result


def recover_label(pred_var, gold_var, l2i_dic, i2l_dic):
    assert len(pred_var) == len(gold_var)
    pred_variable = []
    gold_variable = []
    for i in range(len(gold_var)):
        start_index = gold_var[i].index(l2i_dic['<start>'])
        end_index = gold_var[i].index(l2i_dic['<eos>'])
        pred_variable.append(pred_var[i][start_index:end_index])
        gold_variable.append(gold_var[i][start_index:end_index])

    pred_label = []
    gold_label = []
    for j in range(len(gold_variable)):
        pred_label.append([i2l_dic[t] for t in pred_variable[j]])
        gold_label.append([i2l_dic[t] for t in gold_variable[j]])

    return pred_label, gold_label


class SegmenterEvaluation():
    def evaluate(self, original_labels, predict_labels):
        right, predict = self.get_order(original_labels, predict_labels)
        print('right, predict: ', right, predict)
        right_count = self.rightCount(right, predict)
        if right_count == 0:
            recall = 0
            precision = 0
            f1 = 0
            error = 1
        else:
            recall = right_count / len(right)
            precision = right_count / len(predict)
            f1 = (2 * recall * precision) / (precision + recall)
            error = (len(predict) - right_count) / len(right)
        return precision, recall, f1, error, right, predict

    def rightCount(self, rightList, predictList):
        count = set(rightList) & set(predictList)
        return len(count)

    def get_order(self, original_labels, predict_labels):

        assert len(original_labels) == len(predict_labels)

        start = 1
        end = len(original_labels) - 1  # 当 len(original_labels) -1 > 1的时候,只要有一个字就没问题
        # 按照origin的长度，且删去开头结尾符
        original_labels = original_labels[start:end]
        predict_labels = predict_labels[start:end]

        def merge(labelList):
            # 输入标签字符串，返回一个个词的(begin,end+1)元组
            new_label = []
            chars = ""
            for i, label in enumerate(labelList):
                if label not in ("B", "M", "E", "S"):  # 可能是其他标签
                    if len(chars) != 0:
                        new_label.append(chars)
                    new_label.append(label)
                    chars = ""
                elif label == "B":
                    if len(chars) != 0:
                        new_label.append(chars)
                    chars = "B"
                elif label == "M":
                    chars += "M"
                elif label == "S":
                    if len(chars) != 0:
                        new_label.append(chars)
                    new_label.append("S")
                    chars = ""
                else:
                    new_label.append(chars + "E")
                    chars = ""
            if len(chars) != 0:
                new_label.append(chars)
            orderList = []
            start = 0
            end = 0
            for each in new_label:
                end = start + len(each)
                orderList.append((start, end))
                start = end

            return orderList

        right = merge(original_labels)
        predict = merge(predict_labels)
        return right, predict


def get_f1(gold_label, pred_label):
    assert len(gold_label) == len(pred_label)
    sege = SegmenterEvaluation()
    total_right = 0
    total_pred = 0
    total_gold = 0
    for i in range(len(gold_label)):
        temp_gold, temp_predict = sege.get_order(gold_label[i], pred_label[i])
        temp_right = sege.rightCount(temp_gold, temp_predict)
        total_right += temp_right
        total_gold += len(temp_gold)
        total_pred += len(temp_predict)
    recall = total_right / total_gold
    precision = total_right / total_pred
    f1 = (2 * recall * precision) / (precision + recall)
    return precision, recall, f1


def save_model(path, model, epoch):
    pass


def load_model(path, model):
    return model


if __name__ == "__main__":
    pass


