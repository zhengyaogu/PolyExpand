'''
Code in this file is based on this tutorial:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

'''

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from data import MAX_LEN


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, src_mask):
        '''
        input: a N * L long tensor
        src_mask: a N * L * H tensor

        '''
        embedded = self.embedding(input)
        embedded = embedded * src_mask
        hidden_states, last_hidden = self.gru(embedded)
        return hidden_states, last_hidden


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LEN):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, src_mask):
        '''
        N is the batch size, H is the hidden size
        input:
            input: a 1-d tensor of length N
            hidden: a N * H tensor
            encoder_outputs: a N * L * H tensor
            src_mask: a N * L * H tensor
        
        output:
            output: a N * O tensor
            hidden: a N * H tensor

        '''
        encoder_outputs = encoder_outputs * src_mask

        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), 1)), dim=1)
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)

        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output)

        output = F.relu(output)
        hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(hidden), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, lang, d_model=512, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=512, 
                 dropout=0.1, src_max_len=MAX_LEN, tgt_max_len=MAX_LEN+2):
        super().__init__()
        io_size = len(lang)
        self.lang = lang
        self.input_size = io_size
        self.output_size = io_size
        self.d_model = d_model
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

        self.embedding = nn.Embedding(self.input_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=0.1)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model = d_model,
                                                                        nhead = nhead,
                                                                        dim_feedforward = dim_feedforward,
                                                                        dropout = 0.1),
                                             num_layers = num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model = d_model,
                                                                        nhead = nhead,
                                                                        dim_feedforward = dim_feedforward,
                                                                        dropout = 0.1),
                                             num_layers = num_decoder_layers)
        self.linear = nn.Linear(in_features = d_model, out_features = io_size)
    
    def forward(self, src, tgt):
        '''
        input:
            src: [N, S] tensor
            tgt: [N, T] tensor
        output: [N, T, O]
        '''
        tgt_mask = self.square_subsequent_mask(tgt.shape[1])
        tgt_key_padding_mask = self.key_padding_mask(tgt)
        src_key_padding_mask = self.key_padding_mask(src)

        embedded_src = self.embedding(src) # [N, S, E]
        embedded_tgt = self.embedding(tgt) # [N, T, E]

        memory = self.encoder(embedded_src.transpose(0, 1),
                              src_key_padding_mask = src_key_padding_mask)

        output = self.decoder(embedded_tgt.transpose(0, 1),
                              memory,
                              tgt_mask = tgt_mask,
                              tgt_key_padding_mask = tgt_key_padding_mask)
        output = output.transpose(0, 1) #[N, T, E]
        output = self.linear(output) # [N, T, O]
        output = F.log_softmax(output, dim = -1) # [N, T, O]
        return output
    
    def predict(self, src):
        '''
        input:
            src: [N, S] tensor
        output: [N, ]
        '''
        self.eval()

        with torch.no_grad():
            src_key_padding_mask = self.key_padding_mask(src)

            embedded_src = self.embedding(src) # [N, S, E]
        
            memory = self.encoder(embedded_src.transpose(0, 1),
                                src_key_padding_mask = src_key_padding_mask)

            tgt = torch.tensor( [self.lang.id_by_vocab('[SOS]')] * src.shape[0], 
                                dtype = torch.long ).unsqueeze(1)
            
            for _ in range(self.tgt_max_len - 1):

                embedded_tgt = self.embedding(tgt) # [N, i, E]

                tgt_mask = self.square_subsequent_mask(tgt.shape[1])
                tgt_key_padding_mask = self.key_padding_mask(tgt)

                output = self.decoder(embedded_tgt.transpose(0, 1),
                                     memory,
                                     tgt_mask = tgt_mask,
                                     tgt_key_padding_mask = tgt_key_padding_mask) # [N, i, E]
                
                output = output.transpose(0, 1) #[N, i, E]
                output = self.linear(output) # [N, i, O]
                output = F.log_softmax(output, dim = -1) # [N, i, O]
                output = output.argmax(dim=-1)[:, -1].unsqueeze(1) # [N, 1]
                tgt = torch.cat([tgt, output], dim = 1)

        self.train()
        return tgt.detach()


    def square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def key_padding_mask(self, batch):
        '''
        input:
            batch: [N, L, E] tensor

        '''
        return (batch == self.lang.id_by_vocab('[PAD]'))
