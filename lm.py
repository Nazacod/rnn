import numpy as np
import codecs
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)

START_TOKEN = '<start>'
EOS_TOKEN = '<eos>'
np.random.seed(42)

config = {'lr': 0.08, 'lr_decay': 0.9,
          'max_grad_norm': 5, 'emb_size': 256,
          'hidden_size': 256, 'max_epoch': 10,
          'max_max_epoch': 30, 'batch_size': 64,
          'num_steps': 35, 'vocab_size': 10000,
          'dropout_rate': 0.8}

# config = {'lr': 0.01, 'lr_decay': 0.9,
#           'max_grad_norm': 5, 'emb_size': 256,
#           'hidden_size': 256, 'max_epoch': 9,
#           'max_max_epoch': 30, 'batch_size': 64,
#           'num_steps': 100, 'vocab_size': 10000,
#           'dropout_rate': 0.8}

# config = {'lr': 0.01, 'lr_decay': 0.9,
#           'max_grad_norm': 5, 'emb_size': 256,
#           'hidden_size': 256, 'max_epoch': 6,
#           'max_max_epoch': 13, 'batch_size': 64,
#           'num_steps': 35, 'vocab_size': 10000,
#           'dropout_rate': 0.8}


def normalize(x):
    return x / x.sum(axis=-1)


def batch_generator(data_path, batch_size, num_steps, word_to_id):
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        tokens = f.read().replace("\n", "<eos>").split()

    x = tokens[:len(tokens) - (len(tokens) % (batch_size * num_steps))]
    y = x[1:] + [EOS_TOKEN]
    x = torch.tensor([word_to_id[word] for word in x], dtype=torch.int64).reshape((batch_size, -1))
    y = torch.tensor([word_to_id[word] for word in y], dtype=torch.int64).reshape((batch_size, -1))

    for i in range(x.shape[1] // num_steps):
        x_ = x[:, i * num_steps:(i + 1) * num_steps]
        y_ = y[:, i * num_steps:(i + 1) * num_steps]
        yield x_, y_


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


def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def run_epoch(lr, model, word_to_id, loss_fn, path, optimizer=None, device=None, batch_size=1, num_steps=35):
    total_loss, total_examples = 0.0, 0
    # generator = batch_generator("PTB/ptb.train.txt", batch_size, num_steps, word_to_id)
    generator = batch_generator(path, batch_size, num_steps, word_to_id)
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


def train(token_list, word_to_id, id_to_word):
    """
    Trains n-gram language model on the given train set represented as a list of token ids.
    :param token_list: a list of token ids
    :return: learnt parameters, or any object you like (it will be passed to the next_proba_gen function)
    """
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PTBLM(config["emb_size"], config["hidden_size"],
                  config["vocab_size"])
    print(device)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # model
    # plot_data = []

    for i in range(config['max_max_epoch']):
        lr_decay = config['lr_decay'] ** max(i + 1 - config['max_epoch'], 0.0)
        decayed_lr = config['lr'] * lr_decay

        model.train()
        train_perplexity = run_epoch(decayed_lr, model,
                                     word_to_id, loss_fn,
                                     path="PTB/ptb.train.txt",
                                     optimizer=optimizer,
                                     device=device,
                                     batch_size=config["batch_size"],
                                     num_steps=config['num_steps'])

        model.eval()
        with torch.no_grad():
            dev_perplexity = run_epoch(decayed_lr, model,
                                       word_to_id, loss_fn,
                                       path="PTB/ptb.valid.txt",
                                       device=device,
                                       batch_size=config["batch_size"],
                                       num_steps=config['num_steps'])

        # plot_data.append((i, train_perplexity, decayed_lr))
        print(f'Epoch: {i + 1}. Learning rate: {decayed_lr:.3f}. '
              f'Train Perplexity: {train_perplexity:.3f}. '
              f'Dev Perplexity: {dev_perplexity:.3f}. ')
    with torch.no_grad():
        strings = ancestral_sampling(model, word_to_id, id_to_word, 10, 20, device)
        for string in strings:
            print(string)
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    list_hx = hidden_state
    params.eval()
    for X in token_gen:
        X = torch.tensor([X]).T
        X = X.to(device)
        with torch.no_grad():
            # print(X.shape)
            probs, list_hx = params(X, list_hx)
            probs = probs.transpose(0, 1).contiguous()
            res = probs[0]
            if torch.cuda.is_available():
                res = F.softmax(res, dim=1)
                res = res.to("cpu")
        yield np.array(res), list_hx


def ancestral_sampling(model,
                      word_to_id,
                      id_to_word,
                      size,
                      max_len,
                      device,
                      temperature=1.0):

    # unk_id = word_to_id['<unk>']

    strings = []
    for _ in range(size):
        prev_idx = np.random.choice(len(id_to_word))
        string = [id_to_word[int(prev_idx)]]
        model.eval()
        with torch.no_grad():
            hidden_state = None
            for i in range(max_len):
                X = torch.tensor([[prev_idx]], dtype=torch.int64, device=device)
                logits, hidden_state = model(X, hidden_state)
                softmax = F.softmax(logits, -1).cpu().numpy()[0, 0]

                if temperature != 1.0:
                    softmax = np.float_power(softmax, 1.0 / temperature)
                    softmax /= softmax.sum()

                prev_idx = np.random.choice(list(range(len(softmax))), p=softmax)

                # selection of the most probable word
                # prev_idx = np.argmax(softmax, axis=2)[0, 0]
                string.append(id_to_word[int(prev_idx)])

        strings.append(' '.join(string).split(EOS_TOKEN)[0])
    return strings
