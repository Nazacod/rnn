from typing import List, Any
from random import random
import numpy as np
import torch.nn as nn
import torch
import collections
from string import printable, punctuation, whitespace, ascii_uppercase


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
useful_symbols = []


config = {'lr': 0.0075, 'lr_decay': 0.9,
          'max_grad_norm': 5, 'emb_size': 256,
          'hidden_size': 256, 'max_epoch': 9,
          'max_max_epoch': 50, 'batch_size': 64,
          'num_steps': 35, 'vocab_size': 10000,
          'dropout_rate': 0.9}


def preprocessing(texts, symbols, punc):
    i = 0
    for text in texts:
        texts[i] = text.replace('<br /><br />', '').lower()
        i = i + 1
    i = 0
    for text in texts:
        j = 0
        for symbol in text:
            try:
                symbols.index(symbol)
            except ValueError:
                if symbol in punc:
                    if j > 0 and texts[i][j - 1] == ' ':
                        # texts[i] = texts[i][:j] + ' ' + symbol + ' ' + texts[i][j+1:]
                        texts[i] = ' '.join((texts[i][:j], symbol, texts[i][j + 1:]))
                        j = j + 3
                    else:
                        j = j + 1
                else:
                    # texts[i] = texts[i][:j] + ' ' + symbol + ' ' + texts[i][j + 1:]
                    texts[i] = ' '.join((texts[i][:j], symbol, texts[i][j + 1:]))
                    j = j + 3
            else:
                j = j + 1
        i = i + 1


def tokenization(texts):
    i = 0
    for text in texts:
        texts[i] = text.split()
        i = i + 1


def part_to_tensor(single_part, device, word_to_id=None):
    flag = False
    preprocessing(single_part, useful_symbols, set(punctuation))
    tokenization(single_part)
    max_len = 0
    for text in single_part:
        max_len = max(max_len, len(text))

    if word_to_id is None:
        flag = True
        words = [single_part[i][j] for i in range(len(single_part)) for j in range(len(single_part[i]))]
        freq = 5
        voc = set()
        counter = collections.Counter(words)
        for key in counter.keys():
            if counter[key] >= freq:
                voc.add(key)

        word_to_id = {word: step for step, word in enumerate(voc)}
        word_to_id[PAD_TOKEN] = len(word_to_id)
        word_to_id[UNK_TOKEN] = len(word_to_id)

    for text in single_part:
        for i_word in range(len(text)):
            if text[i_word] not in voc:
                text[i_word] = UNK_TOKEN

    for i in range(len(single_part)):
        single_part[i].extend([PAD_TOKEN] * (max_len - len(single_part[i])))

    tensor = torch.tensor([[word_to_id[word] for word in ex] for ex in single_part],
                            dtype=torch.int64, device=device)

    if flag:
        return tensor, word_to_id
    else:
        return tensor
# def count_labels(labels: List):
#     return {
#         unique_label: sum(1 for label in labels if label == unique_label)
#         for unique_label in set(labels)
#     }


class LSTMCell(nn.Module):
    # input_size = emb_size
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_input = nn.Parameter(torch.Tensor(self.input_size, 4 * self.hidden_size))
        self.B_input = nn.Parameter(torch.Tensor(4 * self.hidden_size))

        self.W_hidden = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.B_hidden = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def forward(self, inp, hx):
        # hx = (h_(t-1), c_(t-1))
        i_all = torch.matmul(inp, self.W_input) + self.B_input
        h_all = torch.matmul(hx[0], self.W_hidden) + self.B_hidden
        list_tensors = torch.chunk(i_all + h_all, 4, dim=1)
        i_t = torch.sigmoid(list_tensors[0])
        f_t = torch.sigmoid(list_tensors[1])
        o_t = torch.sigmoid(list_tensors[2])
        # c_tilde_t = torch.tanh(list_tensors[3])
        c_t = f_t * hx[1] + i_t * (torch.tanh(list_tensors[3]))
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)


# input.shape(batch_size, emb_size)
class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.lstmcell = LSTMCell(self.input_size, self.hidden_size)

    def forward(self, batch_x, hx=None):
        # batch_x.shape = (seq_len, batch_size, emb_size)
        outputs = []
        if self.training:
            mask_h = (torch.rand(batch_x.shape[1], self.hidden_size, device=batch_x.device) < self.dropout_rate)
            mask_c = (torch.rand(batch_x.shape[1], self.hidden_size, device=batch_x.device) < self.dropout_rate)
        if hx is None:
            h_zeros = torch.zeros(batch_x.shape[1], self.hidden_size, device=batch_x.device, dtype=batch_x.dtype)
            c_zeros = torch.zeros(batch_x.shape[1], self.hidden_size, device=batch_x.device, dtype=batch_x.dtype)
            hx = (h_zeros, c_zeros)
        for timestep in range(batch_x.shape[0]):
            hx = self.lstmcell(batch_x[timestep], hx)
            if self.training:
                hx = (hx[0] * mask_h, hx[1] * mask_c)
            outputs.append(hx[0])
        # torch.stack(outputs) = (seq_len, batch_size, hidden_size)
        return torch.stack(outputs), hx


# numHiddenUnits = seq_len(num_steps)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.firstLayer = LSTMLayer(self.input_size, self.hidden_size, self.dropout_rate)
        self.secondLayer = LSTMLayer(self.hidden_size, self.hidden_size, self.dropout_rate)

    def forward(self, batch_x, list_hx=None):
        # batch_x.shape = (seq_len, batch_size, emb_size)
        if self.training:
            mask_1 = (torch.rand(batch_x.shape[1], batch_x.shape[2], device=batch_x.device) < self.dropout_rate)
            mask_2 = (torch.rand(batch_x.shape[1], self.hidden_size, device=batch_x.device) < self.dropout_rate)
            mask_3 = (torch.rand(batch_x.shape[1], self.hidden_size, device=batch_x.device) < self.dropout_rate)
            batch_x *= mask_1
        if list_hx is None:
            out_first, hx_1 = self.firstLayer(batch_x)
            if self.training:
                # print(out_first.shape, out_first.dtype)
                # print(mask_2.shape, mask_2.dtype)
                out_first *= mask_2
            out_second, hx_2 = self.secondLayer(out_first)
            if self.training:
                out_second *= mask_3
        else:
            out_first, hx_1 = self.firstLayer(batch_x, list_hx[0])
            if self.training:
                out_first *= mask_2
            out_second, hx_2 = self.secondLayer(out_first, list_hx[1])
            if self.training:
                out_second *= mask_3

        return out_second, (hx_1, hx_2)


class Linear(nn.Module):
    def __init__(self, emb_size, vocab_size):
        super(Linear, self).__init__()
        self.emb_size = emb_size
        self.vocab_size = vocab_size

        # self.weights = nn.Parameter(torch.Tensor(self.vocab_size, self.emb_size))
        self.B = nn.Parameter(torch.Tensor(self.vocab_size))

        self.reset_parameters()

    def forward(self, weights, inputs):
        #w = (voc_size, emb_size)
        #input = (seq_len, bs, emb_size)
        outputs = []
        # self.weights = weights
        for timestep in range(inputs.shape[0]):
            out = torch.matmul(inputs[timestep], weights.T) + self.B
            outputs.append(out)

        return torch.stack(outputs)


    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.emb_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)


class PTBLM(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size):
        super(PTBLM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Creating an embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.emb_size)
        # Creating a recurrent cell. For multiple recurrent layers, you need to create the same number of recurrent cells.
        self.lstm = LSTM(self.emb_size, self.hidden_size, config['dropout_rate'])
        # Linear layer for projecting outputs from a recurrent layer into space with vocab_size dimension
        # self.decoder = nn.Linear(in_features=self.hidden_size,
        #                          out_features=self.vocab_size)
        self.linear = Linear(self.emb_size, self.vocab_size)
        # self.
        # Weights initialization
        self.init_weights()

    def forward(self, model_input, list_hx=None):
        # embs.shape = (seq_len, batch_size, emb_size)
        embs = self.embedding(model_input).transpose(0, 1).contiguous()
        outputs, list_hx_out = self.lstm(embs, list_hx)
        # logits = self.decoder(outputs).transpose(0, 1).contiguous()
        logits = self.linear(self.embedding.weight.clone(), outputs).transpose(0, 1).contiguous()

        return logits, list_hx_out

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # self.decoder.weight.data.uniform_(-0.1, 0.1)


def batch_generator(tensor, batch_size, word_to_id):
    #tensor.shape = (cnt_example, max_len)

    raw_pad=torch.tensor([word_to_id[PAD_TOKEN]]*tensor.shape[0],dtype=torch.int64,device=tensor.device).reshape(-1, 1)
    y = torch.cat([tensor[:, 1:], raw_pad], dim=1)
    for i in range(batch_size - tensor.shape[0] % batch_size):
        x_ = tensor[i*batch_size:(i+1)*batch_size]
        y_ = y[i*batch_size:(i+1)*batch_size]

        yield x_, y_


def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def run_epoch(lr, model, word_to_id, loss_fn, input, optimizer=None, device=None, batch_size=1):
    total_loss, total_examples = 0.0, 0
    # generator = batch_generator("PTB/ptb.train.txt", batch_size, num_steps, word_to_id)
    generator = batch_generator(input, batch_size, word_to_id)
    list_hx = None
    for step, (X, Y) in enumerate(generator):
        # print(step)
        X = X.to(device)
        Y = Y.to(device)
        if optimizer is not None:
            optimizer.zero_grad()

        logits, list_hx = model(X, list_hx)

        # print(logits.reshape(-1, model.vocab_size).shape)
        # print(Y.reshape(-1).shape)

        # loss = loss_fn(logits.contiguous().view((-1, model.vocab_size)), Y.contiguous().view(-1))
        loss = loss_fn(logits.view((-1, model.vocab_size)), Y.contiguous().view(-1))
        total_examples += loss.size(0)
        total_loss += loss.sum().item()
        loss = loss.mean()

        if optimizer is not None:
            update_lr(optimizer, lr)
            loss.backward()
            optimizer.step()
        # logits.detach_()
        for tpl in list_hx:
            for state in tpl:
                state.detach_()

    return np.exp(total_loss / total_examples)


def train(
        train_texts: List[str],
        train_labels: List[str],
        pretrain_params: Any = None) -> Any:
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :param pretrain_params: parameters that were learned at the pretrain step
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    # ############################ REPLACE THIS WITH YOUR CODE #############################
    # label2cnt = count_labels(train_labels)  # count labels
    # print('Labels counts:', label2cnt)
    # train_size = sum(label2cnt.values())
    # label2prob = {label: cnt / train_size for label, cnt in label2cnt.items()}  # calculate p(label)
    # print(label2prob)
    # return {'prior': label2prob}  # this dummy classifier learns prior probabilities of labels p(label)
    # ############################ REPLACE THIS WITH YOUR CODE #############################


def pretrain(texts_list: List[List[str]]) -> Any:
    """
    Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.
    :param texts_list: a list of list of texts (str objects), one str per example.
        It might be several sets of texts, for example, train and unlabeled sets.
    :return: learnt parameters, or any object you like (it will be passed to the train function)
    """
    np.random.seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    symbols = set(printable)
    print(len(texts_list[0]))
    for text in texts_list[0]:
        for symbol in text:
            if symbol not in symbols:
                symbols.add(symbol)
    symbols = symbols.difference(set(punctuation + whitespace + ascii_uppercase))
    useful_symbols = sorted(symbols)
    useful_symbols = useful_symbols[2:38] + useful_symbols[62:102]

    texts_list = [texts_list[i][j] for i in range(len(texts_list)) for j in range(len(texts_list[i]))]
    matr, word_to_id = part_to_tensor(texts_list, device=device)

    model = PTBLM(config["emb_size"], config["hidden_size"],
                          len(word_to_id))

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for i in range(config['max_max_epoch']):
        lr_decay = config['lr_decay'] ** max(i + 1 - config['max_epoch'], 0.0)
        decayed_lr = config['lr'] * lr_decay

        model.train()
        train_perplexity = run_epoch(decayed_lr, model,
                                     word_to_id, loss_fn,
                                     input=matr,
                                     optimizer=optimizer,
                                     device=device,
                                     batch_size=config["batch_size"])

        print(f'Epoch: {i + 1}. Learning rate: {decayed_lr:.3f}. '
              f'Train Perplexity: {train_perplexity:.3f}. ')


    # print(useful_symbols)
    # for texts in texts_list:
    #     preprocessing(texts, useful_symbols, set(punctuation))
    #     tokenization(texts)
    # max_len = 0
    # print(len(texts_list))
    # for texts in texts_list:
    #     for text in texts:
    #         max_len = max(max_len, len(text))

    # a = [texts_list[i][j][k] for i in range(len(texts_list)) for j in range(len(texts_list[i])) for k in
    #      range(len(texts_list[i][j]))]
    # freq = 5
    # voc = set()
    # counter = collections.Counter(a)
    # for key in counter.keys():
    #     if counter[key] >= freq:
    #         voc.add(key)
    # for part in texts_list:
    #     for text in part:
    #         for i_word in range(len(text)):
    #             if text[i_word] not in voc:
    #                 text[i_word] = UNK_TOKEN
    # #text_list = заменены слова на <unk>  voc() множество слов
    # for part in texts_list:
    #     for i in range(len(part)):
    #         part[i].extend([PAD_TOKEN] * (max_len - len(part[i])))
    # #     neeed word_to_id
    # texts_list = [texts_list[i][j] for i in range(len(texts_list)) for j in range(len(texts_list[i]))]
    # word_to_id = {word: step for step, word in enumerate(voc)}
    # word_to_id[PAD_TOKEN] = len(word_to_id)
    # word_to_id[UNK_TOKEN] = len(word_to_id)
    #
    # # matr.shape = (cnt_predl, max_len)
    # matr = torch.Tensor([[word_to_id[word] for word in ex] for ex in texts_list],
    #                     dtype=torch.int64, device=device)

    return None


def classify(texts: List[str], params: Any) -> List[str]:
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """
       
    # ############################ REPLACE THIS WITH YOUR CODE #############################
    # def random_label(_label2prob):
    #     rand = random()  # random value in [0.0, 1.0) from uniform distribution
    #     for label, prob in _label2prob.items():
    #         rand -= prob
    #         if rand <= 0:
    #             return label
    #
    # label2prob = params['prior']
    # res = [random_label(label2prob) for _ in texts]  # this dummy classifier returns random labels from p(label)
    # print('Predicted labels counts:')
    # print(count_labels(res))
    # return res
    # ############################ REPLACE THIS WITH YOUR CODE #############################
