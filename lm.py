import numpy as np
from collections import Counter
import codecs
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(42)


START_TOKEN = '<start>'
EOS_TOKEN = '<eos>'
device_glob = ""
np.random.seed(42)


def normalize(x):
    return x / x.sum(axis=-1)


def batch_generator(data_path, batch_size, num_steps, word_to_id):
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        tokens = f.read().replace("\n", "<eos>").split()

    x = tokens[:len(tokens)-(len(tokens) % (batch_size*num_steps))]
    y = x[1:] + [EOS_TOKEN]
    x = torch.tensor([word_to_id[word] for word in x], dtype=torch.int64).reshape((batch_size, -1))
    y = torch.tensor([word_to_id[word] for word in y], dtype=torch.int64).reshape((batch_size, -1))

    for i in range(x.shape[1] // num_steps):
        x_ = x[:, i*num_steps:(i+1)*num_steps]
        y_ = y[:, i*num_steps:(i+1)*num_steps]
        yield x_, y_


class LSTMCell(nn.Module):
    # input_size = emb_size
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_input = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.B_input = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.W_hidden = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.B_hidden = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def forward(self, inp, hx):
        #hx = (h_(t-1), c_(t-1))
        i_all = torch.matmul(inp, self.W_input) + self.B_input
        h_all = torch.matmul(hx[0], self.W_hidden) + self.B_hidden
        # tmp = i_all + h_all
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
    def __init__(self, input_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.cnt = 0
        # self.ListOfCells = {}
        self.lstmcell = LSTMCell(self.input_size, self.hidden_size)

    def forward(self, batch_x, hx=None):
        outputs = []
        if hx is None:
            h_zeros = torch.zeros(batch_x.shape[1], self.hidden_size, device=batch_x.device, dtype=batch_x.dtype)
            c_zeros = torch.zeros(batch_x.shape[1], self.hidden_size, device=batch_x.device, dtype=batch_x.dtype)
            hx = (h_zeros, c_zeros)
        for timestep in range(batch_x.shape[0]):
            hx = self.lstmcell(batch_x[timestep], hx)
            outputs.append(hx[0])
        # torch.stack(outputs) = (seq_len, batch_size, hidden_size)
        return torch.stack(outputs), hx


# numHiddenUnits = seq_len(num_steps)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.firstLayer = LSTMLayer(input_size, hidden_size)
        self.secondLayer = LSTMLayer(hidden_size, hidden_size)

    def forward(self, batch_x, list_hx=None):
        if list_hx is None:
            out_first = self.firstLayer(batch_x)
            out_second = self.secondLayer(out_first[0])
        else:
            out_first = self.firstLayer(batch_x, list_hx[0])
            out_second = self.secondLayer(out_first[0], list_hx[1])

        return out_second[0], (out_first[1], out_second[1])


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
        self.lstm = LSTM(self.emb_size, self.hidden_size)
        # Linear layer for projecting outputs from a recurrent layer into space with vocab_size dimension
        self.decoder = nn.Linear(in_features=self.hidden_size,
                                 out_features=self.vocab_size)

        # Weights initialization
        self.init_weights()

    def forward(self, model_input, list_hx=None):
        #embs.shape = (seq_len, batch_size, emb_size)
        embs = self.embedding(model_input).transpose(0, 1).contiguous()
        outputs, list_hx_out = self.lstm(embs, list_hx)
        logits = self.decoder(outputs).transpose(0, 1).contiguous()

        return logits, list_hx_out

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.decoder.weight.data.uniform_(-0.1, 0.1)


def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def run_epoch(lr, model, data, word_to_id, loss_fn, optimizer=None, device=None, batch_size=1, num_steps=35):
    total_loss, total_examples = 0.0, 0
    generator = batch_generator("PTB/ptb.train.txt", batch_size, num_steps, word_to_id)
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


def get_small_config():
    config = {'lr': 0.01, 'lr_decay': 0.9,
              'max_grad_norm': 5, 'emb_size': 256,
              'hidden_size': 256, 'max_epoch': 6,
              'max_max_epoch': 13, 'batch_size': 64,
              'num_steps': 35, 'num_layers': 2,
              'vocab_size': 10000}
    # vocab_size = 10000 + <eos>
    return config


def train(token_list, word_to_id, id_to_word):
    """
    Trains n-gram language model on the given train set represented as a list of token ids.
    :param token_list: a list of token ids
    :return: learnt parameters, or any object you like (it will be passed to the next_proba_gen function)
    """
    torch.manual_seed(42)
    np.random.seed(42)
    # word_to_id[EOS_TOKEN] = len(word_to_id)
    # id_to_word[len(word_to_id)-1] = EOS_TOKEN
    # for step, (x, y) in enumerate(batch_generator("PTB/ptb.train.txt", 2, 35, word_to_id)):
    #     print(step)
    #     print(x, end='\n\n')
    #     print(y, end='\n\n')
    # x, y = next(batch_generator("PTB/ptb.train.txt", 2, 35, word_to_id))
    config = get_small_config()

    # print(len(token_list))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PTBLM(config["emb_size"], config["hidden_size"],
                  config["vocab_size"])
    print(device)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # model
    # plot_data = []
    model.train()

    for i in range(config['max_max_epoch']):
        lr_decay = config['lr_decay'] ** max(i + 1 - config['max_epoch'], 0.0)
        decayed_lr = config['lr'] * lr_decay


        train_perplexity = run_epoch(decayed_lr, model, token_list,
                                     word_to_id, loss_fn,
                                     optimizer=optimizer,
                                     device=device,
                                     batch_size=config["batch_size"],
                                     num_steps=config['num_steps'])

        # plot_data.append((i, train_perplexity, decayed_lr))
        print(f'Epoch: {i + 1}. Learning rate: {decayed_lr:.3f}. '
              f'Train Perplexity: {train_perplexity:.3f}. ')
    # epochs, ppl_train, lr = zip(*plot_data)
    # plt.plot(epochs, ppl_train, 'g', label='Perplexity')
    # plt.savefig('lr.png', dpi=1000, format='png')

    return model


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

    # flag = 0
    list_hx = hidden_state
    for X in token_gen:
        # if flag == 0:
        #     flag = 1
        #     init = params.init_hidden(batch_size=X.size)
        #     pack_hidden = torch.stack([init, init])
        #     pack_hidden_c = pack_hidden
        #     pack_hidden = pack_hidden.to(device)
        #     pack_hidden_c = pack_hidden_c.to(device)
        # print(type(X))
        # print(X.shape)
        X = torch.tensor([X]).T
        X = X.to(device)
        params.eval()
        # h2, h2_c, h1, h1_c
        with torch.no_grad():
            # print(X.shape)
            probs, list_hx = params(X, list_hx)
            probs = probs.transpose(0, 1).contiguous()
            res = probs[0]
            # print(probs.shape)
            if torch.cuda.is_available():
                res = F.softmax(res, dim=1)
                res = res.to("cpu")
            # hidden_state
            # print(res.shape)
        yield np.array(res), list_hx
