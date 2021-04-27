import torch
import time
import math
import json
import random
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from data import MAX_LEN, train_eval_test_split, Lang, PolyDataset
from data import Tokenize, IDize, Pad, ToTensor
from model import EncoderRNN, AttnDecoderRNN, TransformerModel
from torch.utils.data import DataLoader
from torchvision import transforms



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def decode(lang, input_batch, src_mask, encoder, decoder, max_length=MAX_LEN):
    '''
    Takes a bathch and a Seq2Seq model,
    input_batch: N * L
    return a N * L tensor of predicted IDs
    
    '''
    
    encoder_hidden_states, encoder_last_hidden = encoder(input_batch, src_mask)

    decoder_input = torch.tensor([lang.id_by_vocab('[SOS]')] * input_batch.shape[0], dtype=torch.long)

    decoder_hidden = encoder_last_hidden.squeeze(0)

    preds = []
    for di in range(max_length + 1): # +1 for the extra [EOS] token
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_hidden_states, src_mask)

        maxv, maxi = decoder_output.max(dim=1) # decoder_output is a N * O tensor, where O is the output size
        decoder_input = maxi.detach() # 1-d tensor of size N
        preds.append(maxi)
    
    preds = torch.stack(preds, dim=1)

    return preds

def evaluate(lang, loader, model):
    n_correct = 0.
    n_total = 0.
    for batch in tqdm(loader, total=len(loader)):
        src_batch = batch['src']
        tgt_batch = batch['tgt']

        pred = model.predict(src_batch) #[N, T] (including [SOS] and [EOS])

        #print('predicted', pred)
        #print('gold', tgt_batch)

        n_correct += (pred[:, 1:] == tgt_batch[:, 1:]).all(dim=-1).sum().item()
        n_total += 1

    prec = n_correct / n_total
    print('Precision: {:4f}'.format(prec))
    return prec

def train_batch(lang, src_batch, tgt_batch,
                model, optimizer,
                criterion, max_length=MAX_LEN):
    '''
    N is the batch size, L is the sequence length
    src_batch: N * L tensor
    tgt_batch: N* (L + 2) tensor (extra '[EOS]' and '[SOS]' token)
    max_length: the maximum number of tokens in the original dataset

    '''
    model.train()

    optimizer.zero_grad()

    loss = 0

    tgt_batch_y = tgt_batch[:, 1:]
    tgt_batch_input = tgt_batch[:, :-1]

    output = model(src_batch, tgt_batch_input) # [N, T, O]
    output = output.view(-1, output.shape[2]) # [N * T, O]

    targets = tgt_batch_y.flatten()
    
    loss = criterion(output, targets)
    
    loss.backward()

    optimizer.step()

    return loss.item()


def train_epochs(lang, model, n_epochs, train_loader, eval_loader, max_length=MAX_LEN, learning_rate=0.01):
    start = time.time()
    eval_precs = []
    plot_losses = []
    running_loss = 0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for e in range(n_epochs):
        epoch_loss = 0.
        i = 0
        for batch in tqdm(train_loader, total=len(train_loader)):
            src_batch = batch['src']
            tgt_batch = batch['tgt']
            loss = train_batch(lang, src_batch, tgt_batch, model, optimizer, criterion, max_length=MAX_LEN)
            epoch_loss += loss
            running_loss += loss

            if i % 100 == 0:
                print('loss {:.4f}'.format(loss))

            i += 1

        eval_prec = evaluate(lang, eval_loader, encoder, decoder)

        print('Epoch {}: training loss = {:.4f}'.format(e, epoch_loss)
              + '\nEvaluation precision = {:.4f}'.format(eval_prec))
        plot_losses.append(epoch_loss)
    
    return encoder, decoder
        

    showPlot(plot_losses)


def train(data_path, lang_path, n_epochs, batch_size, hidden_size, learning_rate=0.01):
    lang = Lang.read_from_json(lang_path)
    with open(data_path, 'r') as f:
        data = json.load(f)
    train_data, eval_data, test_data = train_eval_test_split(data, ratio=(0.1, 0.1, 0.8))

    transformer = transforms.Compose([Tokenize(),
                                      Pad(),
                                      IDize(lang),
                                      ToTensor()
                                    ])

    train_ds = PolyDataset(train_data, transform=transformer)
    eval_ds = PolyDataset(eval_data, transform=transformer)
    test_ds = PolyDataset(test_data, transform=transformer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    '''
    encoder = EncoderRNN(len(lang), hidden_size)
    decoder = AttnDecoderRNN(hidden_size, len(lang))
    '''

    model = TransformerModel(lang, d_model=512,
                             nhead=8,
                             num_encoder_layers=3,
                             num_decoder_layers=3,
                             dim_feedforward=512,
                             dropout=0.1)

    train_epochs(lang, model, n_epochs, train_loader, eval_loader, max_length=29, learning_rate=0.01)


if __name__ == "__main__":
    train(data_path = 'data/data.json',
          lang_path = 'data/lang.json',
          n_epochs=10,
          batch_size = 64,
          hidden_size = 32,
          learning_rate=0.01)