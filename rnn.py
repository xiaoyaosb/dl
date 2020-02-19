import torch
import torch.nn as nn
import time
import math
import sys
import d2lzh as d2l
from mxnet import nd
import random
import zipfile


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with zipfile.ZipFile(r'D:\python37\新建文件夹\boyu\jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')

# print(corpus_chars[:40])

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:10000]

# print(corpus_chars)

# 我们将每个字符映射成⼀个从0开始的连续整数，⼜称索引，来⽅便之后的数据处理。
# 为了得到索引，我们将数据集⾥所有不同字符取出来，然后将其逐⼀映射到索引来构造词典。
# 接着，打印vocab_size，即词典中不同字符的个数，⼜称词典⼤小。
print(set(corpus_chars))
idx_to_char = list(set(corpus_chars))  # set进行去重，再将去重过后的set(corpus_chars)准变为list
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])  # 构建字典enumerate函数将字符与下标构建成二元组
vocab_size = len(char_to_idx)

# print(vocab_size)#共有1027个不同的字
# print(char_to_idx)#构建的关于corpus_chars中字的字典

corpus_indices = [char_to_idx[char] for char in corpus_chars]  # corpus_indices存的是歌词中（corpus_chars）
# 的字的索引

corpus_dict = dict([char, char_to_idx[char]] for char in corpus_chars)  # 歌词对应的字典
# print(corpus_dict)
# print(corpus_indices)


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None): # 减1是因为输出的索引是相应输⼊的索引加1
    num_examples = (len(corpus_indices) - 1) // num_steps  # //是
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        # yield nd.array(X, ctx), nd.array(Y, ctx)
        yield nd.array(X), nd.array(Y)
'''
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
'''

def one_hot(x, n_class, dtype=torch.float32):
    result = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)  # shape: (n, n_class)
    result.scatter_(1, x.long().view(-1, 1), 1)  # result[i, x[i, 0]] = 1
    return result


x = torch.tensor([0, 2])
x_one_hot = one_hot(x, vocab_size)
#print(x_one_hot)
#print(x_one_hot.shape)


def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

X = torch.arange(10).view(2, 5)
inputs = to_onehot(X, vocab_size)
#print(len(inputs), inputs[0].shape)

#初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
# num_inputs: d
# num_hiddens: h, 隐藏单元的个数是超参数
# num_outputs: q

def get_params():
    def _one(shape):
        param = torch.zeros(shape, device=device, dtype=torch.float32)
        nn.init.normal_(param, 0, 0.01)
        return torch.nn.Parameter(param)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device))
    return (W_xh, W_hh, b_h, W_hq, b_q)

# 定义模型
# 函数rnn用循环的方式依次完成循环神经网络每个时间步的计算。
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


# 函数init_rnn_state初始化隐藏变量，这里的返回值是一个元组。
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

print(X.shape)
print(num_hiddens)
print(vocab_size)
state = init_rnn_state(X.shape[0], num_hiddens, device)
inputs = to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(inputs), inputs[0].shape)
print(len(outputs), outputs[0].shape)
print(len(state), state[0].shape)
print(len(state_new), state_new[0].shape)

#裁剪梯度 循环神经网络中较容易出现梯度衰减或梯度爆炸，
# 这会导致网络几乎无法训练。裁剪梯度（clip gradient）是一种应对梯度爆炸的方法。
# 假设我们把所有模型参数的梯度拼接成一个向量

def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


#定义预测函数
#以下函数基于前缀prefix（含有数个字符的字符串）来预测接下来的num_chars个字符。
# 这个函数稍显复杂，其中我们将循环神经单元rnn设置成了函数参数，
# 这样在后面小节介绍其他循环神经网络时能重复使用这个函数。


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]   # output记录prefix加上预测的num_chars个字符
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(Y[0].argmax(dim=1).item())
    return ''.join([idx_to_char[i] for i in output])

#我们先测试一下predict_rnn函数。
# 我们将根据前缀“分开”创作长度为10个字符（不考虑前缀长度）的一段歌词。
# 因为模型参数为随机值，所以预测结果也是随机的。
#print(predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size, idx_to_char, char_to_idx))

#困惑度
#我们通常使用困惑度（perplexity）来评价语言模型的好坏。回忆一下“softmax回归”一节中交叉熵损失函数的定义。困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，
#最佳情况下，模型总是把标签类别的概率预测为1，此时困惑度为1；
#最坏情况下，模型总是把标签类别的概率预测为0，此时困惑度为正无穷；
#基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数。
#显然，任何一个有效模型的困惑度必须小于类别个数。在本例中，困惑度必须小于词典大小vocab_size。

#定义模型训练函数
#跟之前章节的模型训练函数相比，这里的模型训练函数有以下几点不同：
#使用困惑度评价模型。
#在迭代模型参数前裁剪梯度。
#对时序数据采用不同采样方法将导致隐藏状态初始化的不同。


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach_()
            # inputs是num_steps个形状为(batch_size, vocab_size)的矩阵
            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成形状为
            # (num_steps * batch_size,)的向量，这样跟输出的行一一对应
            y = torch.flatten(Y.T)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())

            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                                        num_hiddens, vocab_size, device, idx_to_char, char_to_idx))

# 训练模型并创作歌词
# 现在我们可以训练模型了。
# 首先，设置模型超参数。我们将根据前缀“分开”和“不分开”分别创作长度为50个字符（不考虑前缀长度）的一段歌词。
# 我们每过50个迭代周期便根据当前训练的模型创作一段歌词。


num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

print(device)

'''
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, device, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
'''
