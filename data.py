import json
import torch
import random
import math
import re
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

MAX_LEN = 29

def txt_to_json(txt_path):
    l = []
    with open(txt_path, 'r') as f:
        line = f.readline()
        while line:
            split_line = line[:-1].split("=")
            l.append(split_line)
            line = f.readline()
    return l

def train_val_test_split(data_path, ratio=(0.9, 0.05, 0.05)):
    '''
    data: the data set
    ratio: ratio of the sizes of training, evaluation, and test set

    '''
    data = txt_to_json(data_path)
    random.shuffle(data)
    cutoff_train, cutoff_val = math.ceil(len(data) * ratio[0]), math.ceil(len(data) * (ratio[0] + ratio[1]))
    train, val, test = data[:cutoff_train], data[cutoff_train : cutoff_val], data[cutoff_val:]
    return train, val, test


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
        l = tqdm(l)
        l.set_description(desc='Constructing language: ')
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
    def construct_from_txt(cls, txt_path):
        l = txt_to_json(txt_path)
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

    def __init__(self, device=torch.device('cpu')):
        self.device = device

    def __call__(self, id_pair):
        src_ids, tgt_ids = id_pair['src'], id_pair['tgt']
        src_mask, tgt_mask = id_pair['src_mask'], id_pair['tgt_mask']
        return {'src': torch.tensor(src_ids, dtype=torch.long).to(self.device), 'tgt': torch.tensor(tgt_ids, dtype=torch.long).to(self.device),
                'src_mask': torch.tensor(src_mask).to(self.device), 'tgt_mask': torch.tensor(tgt_mask).to(self.device)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--preprocess', action='store_true',
                        help='Split the data into training, validation, and test set, and store them in json format')
    parser.add_argument('--lang', action='store_true',
                        help='construct the language that\'s going to be used by the model')
    parser.add_argument('--lang_path', type=str,
                        default='data/lang.json',
                        help='where you store the constructed language')
    parser.add_argument('--train_path', type=str,
                        default='data/train.json',
                        help='where you store the training set')
    parser.add_argument('--val_path', type=str,
                        default='data/val.json',
                        help='where you store the validation set')
    parser.add_argument('--test_path', type=str,
                        default='data/test.json',
                        help='where you store the test set')
    parser.add_argument('--split_ratio', nargs=3, type=float,
                        help='A list specifying train-validation-test set ratios, sums up to 1')
    args = parser.parse_args()
    
    if args.lang:
        lang = Lang.construct_from_txt('data/data.txt')
        with open(args.lang_path, 'w') as f:
            json.dump(lang.export(), f)

    if args.preprocess:
        train, val, test = train_val_test_split('data/dataset.txt', ratio=args.split_ratio)
        with open(args.train_path, 'w') as f:
            json.dump(train, f)
        with open(args.test_path, 'w') as f:
            json.dump(test, f)
        with open(args.val_path, 'w') as f:
            json.dump(val, f)



