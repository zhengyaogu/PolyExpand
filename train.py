import torch
import time
import math
import json
import random
import argparse
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from data import MAX_LEN, train_val_test_split, Lang, PolyDataset
from data import Tokenize, IDize, Pad, ToTensor
from model import EncoderRNN, AttnDecoderRNN, TransformerModel
from torch.utils.data import DataLoader
from torchvision import transforms
from util import CUDA, DeviceAction, print_training_info, count_params
from plot import save_plot



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

def evaluate(lang, loader, model, criterion):
    n_correct = 0.
    n_total = 0.
    val_loss = 0.

    loader = tqdm(loader, total=len(loader))
    loader.set_description('Evaluating')
    for batch in loader:
        src_batch = batch['src']
        tgt_batch = batch['tgt']

        pred, pred_vec = model.predict(src_batch) #[N, T] (including [SOS] and [EOS])

        loss = criterion(pred_vec.view(-1, pred_vec.shape[2]), tgt_batch[:, 1:].flatten())
        val_loss += loss.item()

        pad_locs = tgt_batch == lang.id_by_vocab('[SOS]')
        correct_locs = (pred[:, 1:] == tgt_batch[:, 1:]) | pad_locs[:, 1:]

        n_correct += correct_locs.all(dim=-1).sum().cpu().item()
        n_total += src_batch.shape[0]

    prec = n_correct / n_total
    print('Precision: {:4f}'.format(prec))
    print('Validation Loss: {:4f}'.format(val_loss / len(loader)))

    return prec, val_loss / len(loader)

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
    plot_every = 1000
    start = time.time()
    val_precs = []
    training_loss = []
    val_losses = []
    running_loss = 0

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    i = 0
    for e in range(n_epochs):
        print('\n epoch {}'.format(e+1))
        epoch_loss = 0.
        load = tqdm(train_loader, total=len(train_loader))
        load.set_description('Training')
        for batch in load:
            src_batch = batch['src']
            tgt_batch = batch['tgt']
            loss = train_batch(lang, src_batch, tgt_batch, model, optimizer, criterion, max_length=MAX_LEN)
            epoch_loss += loss
            running_loss += loss

            if i % 10 == 0:
                load.set_postfix({'current epoch loss': loss})
            i += 1

            if i % plot_every == 0:
                training_loss.append(loss)
                eval_prec, val_loss = evaluate(lang, eval_loader, model, criterion)
                val_precs.append(eval_prec)
                val_losses.append(val_loss)


        print('Epoch {}: training loss = {:.4f}'.format(e, epoch_loss)
              + '\nEvaluation precision = {:.4f}'.format(eval_prec))

    save_plot(training_loss, val_precs, val_losses, 'loss_prec.png')
    
    return model



def train(args):
    print('Setting up training...')
    lang = Lang.read_from_json(args.lang_path)

    transformer = transforms.Compose([Tokenize(),
                                      Pad(),
                                      IDize(lang),
                                      ToTensor(device=args.device)
                                    ])

    train_ds = PolyDataset.from_json(args.train_path, transform=transformer)
    val_ds = PolyDataset.from_json(args.val_path, transform=transformer)
    test_ds = PolyDataset.from_json(args.test_path, transform=transformer)

    train_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.bsz, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.bsz, shuffle=True, num_workers=0)

    model = TransformerModel.from_args(args, lang)

    print('done\n')

    print_training_info(args)
    print('\n# trainable parameters'.format(count_params(model)))

    trained_model = train_epochs(lang, model, args.n_epochs, train_loader, val_loader, max_length=29, learning_rate=args.lr)
    torch.save(trained_model.state_dict(), args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # trainer args
    parser.add_argument('--lang_path', type = str,
                        default='data/lang.json',
                        help = 'path to where the language is stored')
    parser.add_argument('--train_path', type = str,
                        default='data/train.json',
                        help = 'path to where the training set is stored')
    parser.add_argument('--val_path', type = str, default='data/val.json',
                        help = 'path to where the validation set is stored')
    parser.add_argument('--test_path', type = str,
                        default='data/test.json',
                        help = 'path to where the test set is stored')
    parser.add_argument('--bsz', type=int,
                        default=512,
                        help = 'batch size')
    parser.add_argument('--n_epochs', type=int,
                        default=20,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float,
                        default=5e-4, 
                        help='learning rate')
    parser.add_argument('--model_path', type = str,
                        default='model/trained.pt',
                        help = 'path to where the trained model is saved')

    # model args
    parser.add_argument('--d_model', type=int,
                        default=256,
                        help='hidden size of the model')
    parser.add_argument('--nhead', type=int,
                        default=8,
                        help='number of heads in Transformer')
    parser.add_argument('--enc_layers', type=int,
                        default=3,
                        help='number of layers in the encoder of Transformer')
    parser.add_argument('--dec_layers', type=int,
                        default=3,
                        help='number of layers in the decoder of Transformer')
    parser.add_argument('--dim_ffn', type=int,
                        default=512,
                        help='output size of the fully connected layer in Transformer')
    parser.add_argument('--dropout', type=int,
                        default=0.1,
                        help='dropout probability')

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('-cpu',
                              action='store_const',
                              const=torch.device('cpu'),
                              dest='device',
                              help='train on cpu')
    device_group.add_argument('-gpu',
                              action='store_const',
                              const=CUDA,
                              dest='device',
                              help='train on gpu')

    args = parser.parse_args()
    
    train(args)
