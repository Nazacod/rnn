from typing import List, Any
from string import printable, punctuation, whitespace, ascii_uppercase
from math import sqrt, log
import matplotlib.pyplot as plt
import numpy as np
import codecs

# random.seed(3)
# random.seed(3)
cache_a = []
cache_err = []

### ~20 mins

def count_labels(labels: List):
    return {
        unique_label: sum(1 for label in labels if label == unique_label)
        for unique_label in set(labels)
    }


def importing_words(file):
    with codecs.open(file, 'r', encoding='utf-8') as inp:
        words = [s.strip() for s in inp.read().strip().split('\n')]
    embeddings = {}
    for word in words:
        ind = word.index(' ')
        embeddings[word[:ind]] = np.fromstring(word[ind + 1:], sep=' ')
    return embeddings


def preprocessing(texts, symbols, punc):
    i = 0
    for text in texts:
        texts[i] = text.lower()
        i = i + 1
    i = 0
    for text in texts:
        j = 0
        for symbol in text:
            try:
                symbols.index(symbol)
            except ValueError:
                if symbol in punc:
                    if j > 0 and texts[i][j-1] == ' ':
                        texts[i] = texts[i][:j] + ' ' + symbol + ' ' + texts[i][j+1:]
                        j = j + 3
                    else:
                        j = j + 1
                else:
                    texts[i] = texts[i][:j] + ' ' + symbol + ' ' + texts[i][j + 1:]
                    j = j + 3
            else:
                j = j + 1
        i = i + 1


def tokenization(texts):
    i = 0
    for text in texts:
        texts[i] = text.split()
        i = i + 1


def vectorization(texts, embeddings):
    result = []
    for text in texts:
        vector = np.array([0]*300)
        n = 0
        # i = 0
        for word in text:
            # i = i + 1
            try:
                vector = vector + embeddings[word]
                n = n + 1
            except KeyError:
                pass
        # print("count word = ", i)
        # print("count embed word = ", n)
        if n != 0:
            result.append(vector / n)
        else:
            result.append(vector)
    return result


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


def tanh(x):
    s = 2*sigmoid(2*x) - 1
    return s


def relu(x):
    return np.maximum(x, 0)


def softmax(ZL):
    check(ZL)
    tmp = np.exp(ZL)
    Y_pred = tmp / np.sum(tmp, axis=1, keepdims=True)
    return Y_pred


def init_params(layer_sizes, activation):
    np.random.seed(3)
    res = {}
    for i in range(1, len(layer_sizes)):
        n = layer_sizes[i-1] + 1
        if activation.__name__ == 'relu':
            # res['w'+str(i)] = np.random.randn(n, layer_sizes[i]) * sqrt(2.0/n)
            res['w' + str(i)] = 0.01 * np.random.randn(n, layer_sizes[i])
        else:
            # res['w'+str(i)] = np.random.randn(n, layer_sizes[i]) * sqrt(1.0/n)
            res['w' + str(i)] = 0.01 * np.random.randn(n, layer_sizes[i])
    return res


def init_y(train_labels):
    result = []
    for word in train_labels:
        if word == 'neg':
            result.append([1, 0])
        else:
            result.append([0, 1])
    return np.array(result)


def conc_column_1(matrix):
    height = matrix.shape[0]
    ones = np.array([[1]*height])
    ones = ones.reshape(height, 1)
    result = np.column_stack((ones[:, 0], matrix))
    return result


def fully_connected(a_prev, W, activation):
    # print('shape AL do', a_prev.shape)
    cache_a.append(a_prev)
    # print('shape AL posle', a_prev.shape)
    a_tilde = conc_column_1(a_prev)
    # print('shape tilde AL do', a_tilde.shape)
    # print(a_tilde.shape)
    # print(W.shape)
    # print('shape W do', W.shape)
    z_cur = np.matmul(a_tilde, W)
    # print('shape tilde AL posle', a_tilde.shape)
    # print('shape W posle', W.shape)
    # print('shape z_cur do', z_cur.shape)
    # print((z_cur.shape))
    # print('shape ZL+1', z_cur.shape)
    a_cur = tanh(z_cur)
    # print('shape z_cur posle', z_cur.shape)
    # print('shape a_cur', a_cur.shape)
    # print('----')
    return a_cur


def ffnn(X, params, activation):
    cnt = len(params)
    a_prev = X
    # print(cnt)
    for i in range(1, cnt):
        # print('hello')
        a_prev = fully_connected(a_prev, params['w'+str(i)], activation)
    cache_a.append(a_prev)
    a_tilde = conc_column_1(a_prev)
    # print('shape a_tilde', a_tilde.shape)
    # print('shape w last', params['w'+str(cnt)].shape)
    z_cur = np.matmul(a_tilde, params['w'+str(cnt)])
    # print('end ffnn')
    # print(a_tilde.shape)
    # print(params['w'+str(cnt)].shape)
    return z_cur


def check(matrix):
    for i in range(0, matrix.shape[0]):
        max_in_row = matrix[i].max()
        if max_in_row > 18:
            matrix[i] = matrix[i] - np.array([max_in_row]*matrix.shape[1])


def softmax_crossentropy(ZL, Y):
    check(ZL)
    tmp = np.exp(ZL)
    # print(tmp)
    Y_pred = tmp/np.sum(tmp, axis=1, keepdims=True)
    # print('Y_pred')
    # print(Y_pred[:10])
    ind = Y.argmax(axis=1)
    # print('Y')
    # print(Y[:10])
    # print(ind)
    # print(Y_pred[:20])
    # print('-----')
    CE = 0
    # print(Y_pred)
    # print(Y)
    for i in range(0, Y_pred.shape[0]):
        # print(Y_pred[i][ind[i]])
        CE = CE - log(Y_pred[i][ind[i]])
    # print(Y.shape[0])
    CE = CE/Y.shape[0]
    # if (CE < 0.06):
    #     print(Y_pred[:30])
    #     print('---------')
    #     print(Y[:30])
    err = Y_pred - Y
    cache_err.append(err)
        # print(cache_err[0])
    return CE


def fully_connected_backward(AL, WL__1, activation):
    # print('--------')
    DZL__1 = cache_err.pop()
    # print('DZ[l+1] shape do ', DZL__1.shape)
    # print('WL(l+1) shape do ', WL__1[1:].shape)
    tmp = np.matmul(DZL__1, WL__1[1:].T)
    # print('DZ[l+1] shape posle ', DZL__1.shape)
    # print('WL(l+1) shape posle ', WL__1[1:].shape)
    # print('tmp shape ', tmp.shape)
    # print('shape AL do ', AL.shape)
    der_act = 1 - AL * AL
    # print('shape AL posle ', AL.shape)
    # print('proizv shape ', der_act.shape)
    # der_act = np.minimum(1, AL)
    DZL = tmp * der_act
    # print('shape DZ[l] ', DZL.shape)
    cache_err.append(DZL)
    AL_1 = cache_a[len(cache_a)-1]
    AL_1 = conc_column_1(AL_1)
    # print('A(l-1) ', AL_1.shape)
    # DWL = np.matmul(AL_1.T, DZL)/AL_1.shape[0]
    DWL = np.matmul(AL_1.T, DZL)
    # print('DW(l) shape ', DWL.shape)
    # print('--------')
    return DWL


def ffnn_backward(weights, activation):
    cnt = len(weights)
    grads = {}
    AL_1 = cache_a[len(cache_a) - 1]
    AL_1 = conc_column_1(AL_1)
    # print('back begin')
    # print('shapeA L_1 ', AL_1.shape)
    # print('shape DZL ', cache_err[0].shape)
    # grads['w'+str(cnt)] = np.matmul(AL_1.T, cache_err[0])/AL_1.shape[0]
    grads['w' + str(cnt)] = np.matmul(AL_1.T, cache_err[0])
    # print('w'+str(cnt))
    # print("grad shape ", grads['w'+str(cnt)].shape)
    for i in range(1, cnt):
        # print('w'+str(cnt-i))
        grads['w'+str(cnt-i)] = fully_connected_backward(cache_a.pop(), weights['w'+str(cnt-i+1)], tanh)
    # print('back end')
    return grads


def sgd_step(weights, grads, learning_rate):
    for key in weights.keys():
        # print('------------')
        # print(grads[key])
        # weights[key] = weights[key] - learning_rate*grads[key] - 1e-5*weights[key]
        weights[key] = weights[key] - learning_rate * grads[key]

# def check_grad(X, weights):
#     w1 = {}
#     w2 = {}
#     w1['w1'] = weights['w1'] + 10
#     w2['w1'] = weights['w1'] - 10
#     z1_last = ffnn(X, w1, tanh)
#     z2_last = ffnn(X, w2, tanh)
#     y1_pr = softmax(z1_last)
#     y2_pr = softmax(z2_last)
#     cache_a.clear()
#     cache_err.clear()
#     res = (y1_pr - y2_pr)/20
#     return res


def train_ffnn(Xtrain, Ytrain, layer_sizes, learning_rate, num_epochs, batch_size):
    weights = init_params(layer_sizes, tanh)
    # weights = {}
    # weights["w1"] = np.zeros((51, 2))
    # print(weights['w1'].shape)
    # print(weights['w1'])
    x_vals = []
    y_vals = []
    x = 0
    i = 0
    res_loss = 0
    # for key in weights.keys():
    #     print('key = ', weights[key].shape)
    for i in range(0, num_epochs):
        for j in range(0, Xtrain.shape[0] // batch_size):
            batch = Xtrain[j*batch_size:(j+1)*batch_size]
            # print(batch)
            # tmp = check_grad(batch, weights)
            z_last = ffnn(batch, weights, tanh)
            # print(Ytrain[j*batch_size:(j+1)*batch_size].shape)
            CE = softmax_crossentropy(z_last, Ytrain[j*batch_size:(j+1)*batch_size])
            res_loss += CE
            # if CE < 0.06:
                # return weights
            # break
            # if x < 50:
            x_vals.append(x)
            y_vals.append(CE)
            # i = i + 1
            x = x + 1
            # print("CE = ", CE)
            # for elem in cache_a:
            #     print(elem.shape)
            # print('--------')
            # for elem in cache_z:
            #     print(elem.shape)
            # print('--------')
            # for elem in cache_err:
            #     print(elem.shape)
            # break
            # print(weights['w1'][:10])
            # print('do grad')
            grads = ffnn_backward(weights, tanh)
            # print(tmp)
            # print(grads['w1'])
            # print(weights['w1'][:10])
            # print(grads)
            # break
            # for key in weights.keys():
            #     print(key, ':', weights[key].shape)
            # for key in grads.keys():
            #     print(key, ':', grads[key].shape)
            # for key in grads.keys():
            #     print('key: ', grads[key])
            # if x == 5:
            #     break
            # if j == 20:
            #     break
            sgd_step(weights, grads, learning_rate)
            # print(weights)
            # print('----------')
            cache_a.clear()
            cache_err.clear()
    plt.figure(figsize=(10, 4))
    plt.plot(x_vals, y_vals)
    plt.show()
    return {'weights': weights, 'loss': res_loss}

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
    embeddings = importing_words('glove.6B.300d.txt')
    train_texts_copy = list(train_texts)
    symbols = set(printable)
    i = 0
    for text in train_texts_copy:
        i = i + 1
        for symbol in text:
            if symbol not in symbols:
                symbols.add(symbol)
    symbols = symbols.difference(set(punctuation + whitespace + ascii_uppercase))
    useful_symbols = sorted(symbols)
    useful_symbols = useful_symbols[2:38] + useful_symbols[62:102]
    preprocessing(train_texts_copy, useful_symbols, set(punctuation))
    tokenization(train_texts_copy)
    vectors = vectorization(train_texts_copy, embeddings)
    X = np.array(vectors)
    Y = init_y(train_labels)
    layer_sizes = [300, 200, 100, 2]
    # print(X[:3])
    # layer_sizes = [50, 2]
    # print(X.shape[0])
    # dic = {}
    # lrn_rate = np.linspace(5e-2, 5e-3, 20)
    # for parametr in lrn_rate:
    result = train_ffnn(X, Y, layer_sizes, learning_rate=0.009736842105263155, num_epochs=100, batch_size=10)
    #dic[str(parametr)] = result['loss']
    # b = list(dic.items())
    # z = sorted(b, key=lambda s: s[1], reverse=True)
    # print(z)
    # weights = train_ffnn(X, Y, layer_sizes, learning_rate=5e-4, num_epochs=1000, batch_size=1000)
    # print(weights)
    return {'weights': result['weights'], 'embeddings': embeddings, "us_symbols": useful_symbols}


def pretrain(texts_list: List[List[str]]) -> Any:
    """
    Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.
    :param texts_list: a list of list of texts (str objects), one str per example.
        It might be several sets of texts, for example, train and unlabeled sets.
    :return: learnt parameters, or any object you like (it will be passed to the train function)
    """
    # ############################ PUT YOUR CODE HERE #######################################
    return None


def classify(texts: List[str], params: Any) -> List[str]:
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """
    preprocessing(texts, params['us_symbols'], set(punctuation))
    tokenization(texts)
    vectors = vectorization(texts, params['embeddings'])
    X = np.array(vectors)
    # print(X[:3])
    Y_pred = softmax(ffnn(X, params['weights'], tanh))
    res = []
    for i in range(0, Y_pred.shape[0]):
        if Y_pred[i][0] < 0.5:
            res.append("pos")
        else:
            res.append("neg")
    return res
