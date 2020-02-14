# 文本处理
import collections
import re


# 文本读入 读取时按照行数读取，不是按照句子读取
def read_txt():
    with open('lp.txt', 'r') as f:
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines


lines = read_txt()
#print(list(lines))


#进行分词
def tokenize(sentences, token='word'):# 此处的sentences实际上是前面函数read_txt返回的结果
    if token == 'word':
        return [sentence.split(' ') for sentence in sentences] # 按照空格进行分词
    elif token == 'char':
        return [list(sentence) for sentence in sentences] #
    else:
        print('ERROR: unkown token type '+token)

tokens = tokenize(lines)
#print(tokens[0:2])


class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = count_corpus(tokens)  # :
        self.token_freqs = list(counter.items())
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def count_corpus(sentences):
    tokens = [tk for st in sentences for tk in st]
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数


vocab = Vocab(tokens)

for i in range(2, 3):
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])



