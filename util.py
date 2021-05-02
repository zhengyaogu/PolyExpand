import argparse
import torch

CUDA = torch.device('cuda:0') if torch.cuda.is_available() else None

def count_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params

class DeviceAction(argparse.Action):

    device_look_up = {'gpu': CUDA,
                      'cpu': torch.device('cpu')}

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest)

    def __call__(self, parser, namespace, device, option_string=None):
        if device == None:
            raise Exception
        setattr(namespace, self.dest, self.device_look_up[device])

def print_transformer_info(args):
    print('Transformer Info')
    print('d_model = {}'.format(args.d_model))
    print('nhead = {}'.format(args.nhead))
    print('# encoder layers = {}'.format(args.enc_layers))
    print('# decoder layers = {}'.format(args.dec_layers))
    print('feed-forward dim = {}'.format(args.dim_ffn))
    print('droput prob. = {}'.format(args.dropout))
    print('device = {}'.format(args.device))


def print_training_info(args):
    print('Training Info')
    print('batch size = {}'.format(args.bsz))
    print('# epochs = {}'.format(args.n_epochs))
    print('learning rate = {}'.format(args.lr))
    print('')
    print_transformer_info(args)

