import json
import torch
import random
import math
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MAX_LEN = 29

def train_eval_test_split(data, ratio=(0.8, 0.1, 0.1)):
    '''
    data: the data set
    ratio: ratio of the sizes of training, evaluation, and test set

    '''
    random.shuffle(data)
    print()
    cutoff_train, cutoff_eval = math.ceil(len(data) * ratio[0]), math.ceil(len(data) * (ratio[0] + ratio[1]))
    train, evaltn, test = data[:cutoff_train], data[cutoff_train : cutoff_eval], data[cutoff_eval:]
    return train, evaltn, test


class Lang:
    
    def __init__(self, vocab_to_id, id_to_vocab):
        self.vocab_to_id = vocab_to_id
        self.id_to_vocab = id_to_vocab
        self.add_word('[SOS]')
        self.add_word('[EOS]')
        self.add_word('[PAD]')
    
    def __len__(self):
        return len(self.id_to_vocab)
    
    @staticmethod
    def sent_to_words(sent):
        return re.findall(r'sin|cos|tan|\d|\w|\(|\)|\+|-|\*+', sent.strip().lower())
    
    @staticmethod
    def contruct(l):
        '''
        l: a list of 2-tuples of the format (factorized, expanded)

        '''
        vocab_to_id = dict()
        id_to_vocab = list()
        i = 0
        for p in l:
            line = p[0] + p[1]
            words = Lang.sent_to_words(line)
            for w in words:
                if not w in vocab_to_id:
                    vocab_to_id[w] = i
                    id_to_vocab.append(w)
                    i += 1
        return vocab_to_id, id_to_vocab
    
    def add_word(self, word):
        if not word in self.vocab_to_id:
            i = len(self.id_to_vocab)
            self.vocab_to_id[word] = i
            self.id_to_vocab.append(word)
            

    def vocab_by_id(self, i):
        return self.id_to_vocab[i]
    
    def id_by_vocab(self, vocab):
        return self.vocab_to_id[vocab]
    
    def export(self):
        return [self.vocab_to_id, self.id_to_vocab]
    
    @classmethod
    def construct_from_json(cls, json_path):
        with open(json_path, 'r') as f:
            l = json.load(f)
        vocab_to_id, id_to_vocab = Lang.contruct(l)
        return cls(vocab_to_id, id_to_vocab)
    
    @classmethod
    def read_from_json(cls, json_path):
        with open(json_path, 'r') as f:
            vocab_to_id, id_to_vocab = json.load(f)
        return cls(vocab_to_id, id_to_vocab)


class PolyDataset(Dataset):
    
    def __init__(self, data, transform=None):
        '''
        data: a list of 2-tuples of the format (factorized, expanded)

        '''
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        inst = self.data[idx]
        inst = {'src': inst[0], 'tgt': inst[1]}
        if self.transform:
            inst = self.transform(inst)
        return inst
    
    @staticmethod
    def from_json(json_path, transform=None):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return PolyDataset(data, transform=transform)


class Tokenize(object):

    def __call__(self, sent_pair):
        src_sent, tgt_sent = sent_pair['src'], sent_pair['tgt']
        return {'src': list(src_sent), 'tgt': list(tgt_sent)}

class Pad(object):

    def __call__(self, tokens_pair):
        src_tokens, tgt_tokens = tokens_pair['src'], tokens_pair['tgt']
        src_mask = [False] * len(src_tokens) + [True] * (MAX_LEN - len(src_tokens))
        tgt_mask = [False] * (len(tgt_tokens) + 1) + [True] * (MAX_LEN - len(tgt_tokens))
        src_tokens = src_tokens + ['[PAD]'] * (MAX_LEN - len(src_tokens))
        tgt_tokens = ['[SOS]'] + tgt_tokens + ['[EOS]'] + ['[PAD]'] * (MAX_LEN - len(tgt_tokens))
        return {'src': src_tokens, 'tgt': tgt_tokens,
                'src_mask': src_mask, 'tgt_mask': tgt_mask}

class IDize(object):

    def __init__(self, lang):
        self.lang = lang

    def tokens_to_ids(self, tokens):
        ids = []
        for t in tokens:
            ids.append(self.lang.id_by_vocab(t))
        return ids
    
    def __call__(self, padded_tokens_pair):
        src_tokens, tgt_tokens = padded_tokens_pair['src'], padded_tokens_pair['tgt']
        return {'src': self.tokens_to_ids(src_tokens), 'tgt': self.tokens_to_ids(tgt_tokens),
                'src_mask': padded_tokens_pair['src_mask'], 'tgt_mask': padded_tokens_pair['tgt_mask']}

class ToTensor(object):

    def __call__(self, id_pair):
        src_ids, tgt_ids = id_pair['src'], id_pair['tgt']
        src_mask, tgt_mask = id_pair['src_mask'], id_pair['tgt_mask']
        return {'src': torch.tensor(src_ids, dtype=torch.long), 'tgt': torch.tensor(tgt_ids, dtype=torch.long),
                'src_mask': torch.tensor(src_mask), 'tgt_mask': torch.tensor(tgt_mask)}


if __name__ == "__main__":
    lang = Lang.construct_from_json('data/data.json')
    with open('data/lang.json', 'w') as f:
        json.dump(lang.export(), f)

