import argparse
import torch
from train import evaluate
from util import CUDA, count_params
from data import Lang, Tokenize, IDize, Pad, ToTensor, PolyDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from model import TransformerModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # evaluation args
    parser.add_argument('--lang_path', type = str,
                        default='data/lang.json',
                        help = 'path to where the language is stored')
    parser.add_argument('--test_path', type = str,
                        default='data/test.json',
                        help = 'path to where the test set is stored')
    parser.add_argument('--bsz', type=int,
                        default=512,
                        help = 'batch size')
    parser.add_argument('--model_path', type = str,
                        default='model/trained.pt',
                        help = 'where the trained model is stored')

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
    parser.add_argument('--dropout', type=float,
                        default=0.1,
                        help='dropout probability')

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('-cpu',
                              action='store_const',
                              const=torch.device('cpu'),
                              dest='device',
                              help='evaluate on cpu')
    device_group.add_argument('-gpu',
                              action='store_const',
                              const=CUDA,
                              dest='device',
                              help='evaluate on gpu')
    args = parser.parse_args()

    # set up evaluation
    lang = Lang.read_from_json(args.lang_path)

    transformer = transforms.Compose([Tokenize(),
                                      Pad(),
                                      IDize(lang),
                                      ToTensor(device=args.device)
                                      ])
    test_ds = PolyDataset.from_json(args.test_path, transform=transformer)
    test_loader = DataLoader(test_ds, batch_size=args.bsz, shuffle=True, num_workers=0)
    model = TransformerModel.from_args(args, lang)
    model.load_state_dict(torch.load(args.model_path))

    criterion = torch.nn.NLLLoss()

    evaluate(lang, test_loader, model, criterion)

