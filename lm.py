import numpy as np
from collections import Counter
import codecs
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# ##
# import os
# import collections
# from pathlib import Path
# PTB_PATH = Path(__file__).with_name("PTB")
# ##
torch.manual_seed(42)

NGRAM = 2
START_TOKEN = '<start>'
EOS_TOKEN = '<eos>'
device_glob = ""
np.random.seed(42)


def normalize(x):
    return x / x.sum(axis=-1)


def batch_generator(data_path, batch_size, num_steps):
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        tokens = f.read().replace("\n", "<eos>").split()
    counter = Counter(tokens)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    input_indexes = [word_to_id[word] for word in tokens if word in word_to_id]
    size = num_steps * batch_size
    input_indexes = input_indexes[:len(input_indexes) - (len(input_indexes) % (size))]
    #     print([id_to_word[id1] for id1 in input_indexes if id1 in id_to_word])
    # ???? <eos> <start> ????
    for i in range(len(input_indexes) // size):
        x = torch.tensor(input_indexes[i * size:(i + 1) * size]
                         , dtype=torch.int64)
        y = torch.tensor(input_indexes[1 + i * size:(i + 1) * size] + [word_to_id[EOS_TOKEN]]
                         , dtype=torch.int64)
        yield x.view(batch_size, num_steps), y.view(batch_size, num_steps)


class LSTMCell(nn.Module):
    # input_size = emb_size
    def __init__(self, input_size, hidden_size, batch_size, device):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_input = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.B_input = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.W_hidden = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.B_hidden = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()
        self.to(device)

    def forward(self, inp, initial_state, initial_state_c):
        # print(inp.device)
        # print(self.W_input.device)
        # print(self.B_input.device)
        # print(inp.shape)
        i_all = torch.matmul(inp, self.W_input) + self.B_input
        h_all = torch.matmul(initial_state, self.W_hidden) + self.B_hidden
        tmp = i_all + h_all
        list_tensors = torch.chunk(tmp, 4, dim=1)
        i_t = torch.sigmoid(list_tensors[0])
        f_t = torch.sigmoid(list_tensors[1])
        o_t = torch.sigmoid(list_tensors[2])
        c_tilde_t = torch.tanh(list_tensors[3])
        c_t = f_t * initial_state_c + i_t * c_tilde_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)


# input.shape(batch_size, emb_size)
class LSTMLayer(nn.Module):
    def __init__(self, numHiddenUnits, input_size, hidden_size, batch_size, device):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.numHiddenUnits = numHiddenUnits
        self.lstmcell = LSTMCell(input_size, hidden_size, batch_size, device)
        # self.cnt = 0
        # self.ListOfCells = {}
        # for i in range(self.numHiddenUnits):
        #     self.ListOfCells[str(i)] = LSTMCell(input_size, hidden_size, batch_size, device)
        # self.to(device)

    def forward(self, batch_x, initial_state, initial_state_c):
        outputs = []
        h = initial_state
        c = initial_state_c
        # batch_x.shape = (seq_len, batch_size, emb_size)
        # print("LSTMLayer")
        # print(self.ListOfCells.device)
        if len(batch_x.shape) == 3:
            for timestep in range(batch_x.shape[0]):
                h, c = self.lstmcell(batch_x[timestep], h, c)
                # h = result[0]
                # c = result[1]
                outputs.append(h)
        else:
            h, c = self.lstmcell(batch_x, h, c)
            # h = result[0]
            # c = result[1]
            outputs.append(h)
            return h, h, c
        #     result = self.ListOfCells[str(self.cnt)](batch_x, self.hid, self.hid_c)
        #     self.cnt += 1
        #     params = get_small_config()
        #     if self.cnt == params['num_steps'] - 1:
        #         self.cnt = 0
        #     outputs.append(h)
        #     h = result[0]
        #     c = result[1]

        # torch.stack(outputs) = (seq_len, batch_size, hidden_size)
        return torch.stack(outputs), h, c


# numHiddenUnits = seq_len(num_steps)
class LSTM(nn.Module):
    def __init__(self, numHiddenUnits, input_size, hidden_size, batch_size, num_layers, device):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.firstLayer = LSTMLayer(numHiddenUnits, input_size, hidden_size, batch_size, device)
        self.secondLayer = LSTMLayer(numHiddenUnits, hidden_size, hidden_size, batch_size, device)
        self.to(device)
        # self.ListOfLayers = []
        # for i in range(num_layers):
        #     if i == 0:
        #         self.ListOfLayers.append(LSTMLayer(numHiddenUnits, input_size, hidden_size, batch_size))
        #     else:
        #         self.ListOfLayers.append(LSTMLayer(numHiddenUnits, hidden_size, hidden_size, batch_size))

    def forward(self, batch_x, h2, h2_c, h1, h1_c):
        # for i in range(self.num_layers):
        #     if i == 0:
        #         out = self.ListOfLayers[i](batch_x, initial_state, initial_state_c)
        #     else:
        #         out = self.ListOfLayers[i](out[0], out[1], out[2])
        # print("LSTM")
        # print(self.firstLayer.ListOfCells[0].W_input.device)
        out_first = self.firstLayer(batch_x, h1, h1_c)
        out_second = self.secondLayer(out_first[0], h2, h2_c)
        return out_second[0], out_second[1], out_second[2], out_first[1], out_first[2]


class PTBLM(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size, num_steps, batch_size, num_layers, device):
        super(PTBLM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Creating an embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_size)
        # Creating a recurrent cell. For multiple recurrent layers, you need to create the same number of recurrent cells.
        self.lstm = LSTM(num_steps, emb_size, hidden_size, batch_size, num_layers, device)
        # Linear layer for projecting outputs from a recurrent layer into space with vocab_size dimension
        self.decoder = nn.Linear(in_features=hidden_size,
                                 out_features=vocab_size)

        # Weights initialization
        self.init_weights()

    def forward(self, model_input, h2, h2_c, h1, h1_c):
        #embs.shape = (seq_len, batch_size, emb_size)
        # print("PTBLM")
        # print(model_input.shape)
        embs = self.embedding(model_input).transpose(0, 1).contiguous()
        # print('embed!')
        # print(embs.shape)
        outputs, h2, h2_c, h1, h1_c = self.lstm(embs, h2, h2_c, h1, h1_c)
        # print(outputs.shape)
        logits = self.decoder(outputs).transpose(0, 1).contiguous()

        return logits, h2, h2_c, h1, h1_c

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, batch_size: int):
        return torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)


def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr
####
# def _read_words(filename):
#     with open(filename, "r") as f:
#         return f.read().replace("\n", "<eos>").split()
#
#
# def _build_vocab(filename):
#     data = _read_words(filename)
#
#     counter = collections.Counter(data)
#     count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
#
#     words, _ = list(zip(*count_pairs))
#     word_to_id = dict(zip(words, range(len(words))))
#     id_to_word = {v: k for k, v in word_to_id.items()}
#
#     return word_to_id, id_to_word
#
#
# def _file_to_word_ids(filename, word_to_id):
#     data = _read_words(filename)
#     return [word_to_id[word] for word in data if word in word_to_id]
#
#
# def load_dataset(data_path=None):
#     train_path = os.path.join(data_path, "ptb.train.txt")
#     dev_path = os.path.join(data_path, "ptb.valid.txt")
#     test_path = os.path.join(data_path, "ptb.test.txt")
#
#     word_to_id, id_to_word = _build_vocab(train_path)
#     train_data = _file_to_word_ids(train_path, word_to_id)
#     dev_data = _file_to_word_ids(dev_path, word_to_id)
#     test_data = _file_to_word_ids(test_path, word_to_id)
#
#     return train_data, dev_data, test_data, word_to_id, id_to_word
#

#####

def run_epoch(lr, model, data, word_to_id, loss_fn, optimizer=None, device=None, batch_size=1, num_steps=35):
    total_loss, total_examples = 0.0, 0
    generator = batch_generator("PTB/ptb.train.txt", batch_size, num_steps)
    # print('Hello')
    for step, (X, Y) in enumerate(generator):
        print(step)
        # print(X.shape)
        # print(X.device)
        X = X.to(device)
        # print(X.device)
        # print(Y.device)
        Y = Y.to(device)
        # print(Y.device)
        if optimizer is not None:
            optimizer.zero_grad()
        if step == 0:
            init = model.init_hidden(batch_size)
            initial_state = init[0]
            initial_state_c = init[1]
            initial_state = initial_state.to(device)
            h2 = initial_state
            h1 = initial_state
            initial_state_c = initial_state_c.to(device)
            h2_c = initial_state_c
            h1_c = initial_state_c

        logits, h2, h2_c, h1, h1_c = model(X, h2, h2_c, h1, h1_c)

        # print(logits.shape)
        loss = loss_fn(logits.view((-1, model.vocab_size)), Y.view(-1))
        logits.detach_()
        h2.detach_()
        h2_c.detach_()
        h1.detach_()
        h1_c.detach_()
        total_examples += loss.size(0)
        total_loss += loss.sum().item()
        loss = loss.mean()

        if optimizer is not None:
            update_lr(optimizer, lr)
            loss.backward()
            optimizer.step()

    return np.exp(total_loss / total_examples)


def get_small_config():
    config = {'lr': 0.1, 'lr_decay': 0.5,
              'max_grad_norm': 5, 'emb_size': 200,
              'hidden_size': 200, 'max_epoch': 5,
              'max_max_epoch': 13, 'batch_size': 64,
              'num_steps': 35, 'num_layers': 2,
              'vocab_size': 10000}
    return config


# def batch_for_x(input_indexes, batch_size, num_steps):
#     size = num_steps * batch_size
#     length = len(input_indexes)
#     input_indexes = input_indexes[:length - (length % (size))]
#     for i in range(length // size):
#         x = torch.tensor(input_indexes[i * size:(i + 1) * size]
#                          , dtype=torch.int64)
#         yield x.view(batch_size, num_steps)


def train(token_list, word_to_id, id_to_word):
    """
    Trains n-gram language model on the given train set represented as a list of token ids.
    :param token_list: a list of token ids
    :return: learnt parameters, or any object you like (it will be passed to the next_proba_gen function)
    """
    config = get_small_config()
    # print(len(token_list))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PTBLM(config["emb_size"], config["hidden_size"],
                  config["vocab_size"], config["num_steps"],
                  config["batch_size"], config['num_layers'], device)
    # print(device)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # model
    plot_data = []
    for i in range(config['max_max_epoch']):
        lr_decay = config['lr_decay'] ** max(i + 1 - config['max_epoch'], 0.0)
        decayed_lr = config['lr'] * lr_decay

        model.train()
        train_perplexity = run_epoch(decayed_lr, model, token_list,
                                     word_to_id, loss_fn,
                                     optimizer=optimizer,
                                     device=device,
                                     batch_size=config["batch_size"],
                                     num_steps=config["num_steps"])

        plot_data.append((i, train_perplexity, decayed_lr))
        print(f'Epoch: {i + 1}. Learning rate: {decayed_lr:.3f}. '
              f'Train Perplexity: {train_perplexity:.3f}. ')
    epochs, ppl_train, lr = zip(*plot_data)
    plt.plot(epochs, ppl_train, 'g', label='Perplexity')
    plt.savefig('lr.png', dpi=1000, format='png')
    return model


    ############################# REPLACE THIS WITH YOUR CODE #############################
    l = len(token_list)
    counters = [Counter(tuple(token_list[i:i + ng]) for i in range(l - ng)) for ng in range(1, NGRAM + 1)]
    vocab_size = len(word_to_id)
    return vocab_size, counters
    ############################# REPLACE THIS WITH YOUR CODE #############################


def next_proba_gen(token_gen, params, hidden_state=None):
    """
    For each input token estimate next token probability distribution.
    :param token_gen: generator returning sequence of arrays of token ids (each array has batch_size independent ids);
     i-th element of next array is next token for i-th element of previous array
    :param params: parameters received from train function
    :param hidden_state: the initial state for next token that may be required 
     for sampling from the language model
    :param hidden_state: use this as the initial state for your language model(if it is not None).
     That may be required for sampling from the language model.

    :return: probs: for each array from token_gen should yield vector of shape (batch_size, vocab_size)
     representing predicted probabilities of each token in vocabulary to be next token.
     hidden_state: return the hidden state at each time step of your language model. 
     For sampling from language model it will be used as the initial state for the following tokens.
    """

    # config = get_small_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # X.shape(batch_size, )

    flag = 0
    for X in token_gen:
        if flag == 0:
            flag = 1
            init = params.init_hidden(batch_size=X.size)
            if hidden_state is None:
                hidden_state = init[0]
            else:
                hidden_state = torch.tensor(hidden_state)
            hidden_state_c = init[1]
            hidden_state = hidden_state.to(device)
            h1 = hidden_state
            h2 = hidden_state
            hidden_state_c = hidden_state_c.to(device)
            h1_c = hidden_state_c
            h2_c = hidden_state_c
        # print(type(X))
        # print(X.shape)
        X = torch.tensor(X)
        X = X.to(device)
        params.eval()
        # h2, h2_c, h1, h1_c
        with torch.no_grad():
            probs, h2, h2_c, h1, h1_c = params(X, h2, h2_c, h1, h1_c)
            # print(probs.shape)
            if torch.cuda.is_available():
                probs = F.softmax(probs, dim=1)
                probs = probs.to("cpu")
            # hidden_state
        yield np.array(probs), hidden_state


# ####
# raw_data = load_dataset(PTB_PATH)
# train_data, dev_data, test_data, word_to_id, id_to_word = raw_data
# token_list = train_data
#
# config = get_small_config()
# # print(len(token_list))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = PTBLM(config["emb_size"], config["hidden_size"],
#         config["vocab_size"], config["num_steps"],
#         config["batch_size"], config['num_layers'], device)
# # print(device)
# model.to(device)
# loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
# optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
# # model
# plot_data = []
# for i in range(config['max_max_epoch']):
#     lr_decay = config['lr_decay'] ** max(i + 1 - config['max_epoch'], 0.0)
#     decayed_lr = config['lr'] * lr_decay
#
#     model.train()
#     train_perplexity = run_epoch(decayed_lr, model, token_list,
#                                 word_to_id, loss_fn,
#                                 optimizer=optimizer,
#                                 device=device,
#                                 batch_size=config["batch_size"],
#                                 num_steps=config["num_steps"])
#
#     plot_data.append((i, train_perplexity, decayed_lr))
#     print(f'Epoch: {i + 1}. Learning rate: {decayed_lr:.3f}. '
#         f'Train Perplexity: {train_perplexity:.3f}. ')
# epochs, ppl_train, lr = zip(*plot_data)
# plt.plot(epochs, ppl_train, 'g', label='Perplexity')
# plt.savefig('lr.png', dpi=1000, format='png')