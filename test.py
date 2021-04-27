import torch
from data import Lang
from model import EncoderRNN, AttnDecoderRNN, TransformerModel

def test_encoder_decoder():
    input_size = 31
    hidden_size = 8
    N = 2
    L = 29
    H = hidden_size
    t = torch.randint(low=0, high=30, size=(N, L))
    encoder = EncoderRNN(input_size = input_size, hidden_size = H)
    decoder = AttnDecoderRNN(hidden_size=H, output_size = input_size)

    encoder_hidden_states, encoder_last_hidden = encoder(t)

    decoder_input = torch.randint(low=0, high=30, size=(N,))
    output, hidden, attn_weights = decoder(decoder_input, 
                                           encoder_last_hidden.squeeze(0), 
                                           encoder_hidden_states.transpose(0, 1))
                                    
def test_transformer():
    lang = Lang.read_from_json('data/lang.json')
    T = TransformerModel(lang, 8, 1, 1, 1, 16, 0.1, 5, 7)
    src = torch.tensor([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
    tgt = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2]])
    r = T(src, tgt)
    print(r.shape)

def test_transformer_predict():
    lang = Lang.read_from_json('data/lang.json')
    T = TransformerModel(lang, 8, 1, 1, 1, 16, 0.1, 5, 7)
    src = torch.tensor([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
    r = T.predict(src)
    print(r.shape)
    print(T.tgt_max_len)

if __name__ == "__main__":
    test_transformer_predict()
    