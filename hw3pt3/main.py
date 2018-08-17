import argparse
import time
import os
import csv
from datetime import datetime
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from ctcdecode import CTCBeamDecoder
from warpctc_pytorch import CTCLoss
import Levenshtein as Lev


parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('--data-dir', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--weights-dir', type=str, default='./weights/',
                    help='data directory')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, GRU)')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, #default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=40, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.2,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.5,
                    help='dropout for input layers (0 = no dropout)')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=2222,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')
parser.add_argument('--wdecay', type=float, default=1.2e-6, #default=1.2e-6,
                    help='weight decay applied to all weights')

PHONEME_MAP = [
    ' ', '_', '+', '~', '!', '-', '@', 'a', 'A', 'h', 'o', 'w', 'y', 'b', 'c', 'd', 'D', 'e', 'r', 'E', 'f',
    'g', 'H', 'i', 'I', 'j', 'k', 'l', 'm', 'n', 'G', 'O', 'Y', 'p', 'R', 's', 'S', '.', 't', 'T', 'u', 'U',
    'v', 'W', '?', 'z', 'Z', ]

"""
counters = [0] * 139

def calc_distribution(labels):
    for l in labels:
        for p in l:
            counters[p + 1] += 1

calc_distribution(ytrain)
calc_distribution(ydev)
counters[0] = sum(counters) / 10
normalized = np.array([c / sum(counters) for c in counters])
norm_log = np.log(normalized)
"""
PHONEME_DISTRIBUTION = np.array([
  -2.39789527, -5.99335091, -5.99341193, -5.99341193, -7.67744609, -7.67744609, -7.67744609, -7.23540204, -7.23540204, -7.23540204,
  -6.05643829, -6.05650328, -6.05650328,-10.19238503,-10.19238503, -10.19238503, -6.45616827, -6.45616827, -6.45616827, -5.16378204,
  -5.16378204, -5.16378204, -5.18329565, -5.18332279, -5.18332279, -3.44605889, -3.44606366, -3.44606844, -5.68228521, -5.68228521,
  -5.68228521, -6.71855192, -6.71855192, -6.71855192, -5.49820682, -5.49820682, -5.49820682, -5.28686939, -5.28686939, -5.28686939,
  -6.67045804, -6.67045804, -6.67045804, -4.32413363, -4.32413363, -4.32413363, -5.15084997, -5.15084997, -5.15084997, -4.79014721,
  -4.79014721, -4.79018385, -4.74501707, -4.74501707, -4.74503459, -5.32278892, -5.32278892, -5.32278892, -5.26554382, -5.26554382,
  -5.26557329, -6.21788355, -6.21788355, -6.21788355, -5.7441064 , -5.7441064 , -5.7441064 , -3.98186658, -3.98186658, -3.98186658,
  -4.52422973, -4.52422973, -4.52422973, -6.38599358, -6.38599358, -6.38599358, -4.46021528, -4.46022845, -4.46022845, -4.50039265,
  -4.50040636, -4.50040636, -4.71145821, -4.71147514, -4.71147514, -3.87276526, -3.87276526, -3.87277258, -6.03846853, -6.03846853,
  -6.03853236, -5.56498411, -5.56498411, -5.56498411, -7.51543683, -7.51543683, -7.51543683, -4.82671299, -4.82671299, -4.82671299,
  -4.33125143, -4.33125143, -4.33125143, -4.23291049, -4.23291049, -4.23292098, -5.94214914, -5.94214914, -5.94214914, -4.53554078,
  -4.53564019, -4.53578223, -3.93976082, -3.93976082, -3.93976865, -6.54014426, -6.54014426, -6.54014426, -7.11492292, -7.11492292,
  -7.11492292, -5.61036698, -5.61036698, -5.61036698, -5.25972593, -5.25972593, -5.25972593, -5.44332569, -5.4433609 , -5.4433609 ,
  -5.98957502, -5.98957502, -5.98957502, -4.72671187, -4.72671187, -4.72672907, -8.45063598, -8.45063598, -8.45063598
], dtype=float)


def to_float_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()

def to_long_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).long()

def to_int_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).int()

def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array)

def to_np(x):
    return x.data.cpu().numpy()

def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        if y is not None:
            self.total_phonemes = sum(len(yi) for yi in y)
        else:
            self.total_phonemes = -1

        print("n_utters", self.x.shape[0], "total_phonemes", self.total_phonemes)

    def __getitem__(self, idx):
        frames = self.x[idx]
        return to_float_tensor(frames), \
               to_int_tensor(self.y[idx] + 1 if self.y is not None else np.array([-1]))

    def __len__(self):
        return self.x.shape[0]  # // 10


def my_collate(batch):
    batch_size = len(batch)
    batch = sorted(batch, key=lambda b: b[0].size(0), reverse=True)  # sort the batch by seq_len desc
    max_seq_len = batch[0][0].size(0)
    channels = batch[0][0].size(1)
    pack = torch.zeros(max_seq_len, batch_size, channels)
    all_labels = []
    seq_sizes = []
    label_sizes = torch.zeros(batch_size).int()
    for i, (frames, labels) in enumerate(batch):
        seq_size = frames.size(0)
        seq_sizes.append(seq_size)

        labele_size = labels.size(0)
        label_sizes[i] = labele_size

        pack[:seq_size, i, :] = frames
        all_labels.append(labels)

    return pack, seq_sizes, all_labels, label_sizes


def get_data_loaders(args, use_dev_for_train=False):
    print("loading data")

    xdev = np.load(args.data_dir + '/dev.npy')
    ydev = np.load(args.data_dir + '/dev_subphonemes.npy')
    if use_dev_for_train:
        xtrain = np.load(args.data_dir + '/dev.npy')
        ytrain = np.load(args.data_dir + '/dev_subphonemes.npy')
    else:
        xtrain = np.load(args.data_dir + '/train.npy')
        ytrain = np.load(args.data_dir + '/train_subphonemes.npy')

    print("load complete")
    kwargs = {'num_workers': 3, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        MyDataset(xtrain, ytrain),
        batch_size=args.batch_size, shuffle=True, collate_fn=my_collate, **kwargs)
    dev_loader = torch.utils.data.DataLoader(
        MyDataset(xdev, ydev),
        batch_size=args.batch_size, shuffle=True, collate_fn=my_collate, **kwargs)

    return train_loader, dev_loader


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if dropout == 0 or not self.training:
            return x
        mask = x.data.new(1, x.size(1), x.size(2))
        mask = mask.bernoulli_(1 - dropout)
        mask = Variable(mask, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BnLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, batch_norm=False):
        super(BnLSTM, self).__init__()
        self.batch_norm = SequenceWise(nn.BatchNorm1d(hidden_size * 2)) if batch_norm else None
        self.rnn = rnn_type(input_size, hidden_size, bidirectional=True)

    def forward(self, x, seq_sizes):
        out = pack_padded_sequence(x, seq_sizes)
        out, h = self.rnn(out)
        out, _ = pad_packed_sequence(out)

        if self.batch_norm is not None:
            out = self.batch_norm(out)
        return out, h


class MyModel(nn.Module):
    def __init__(self, nhid, nlayers, dropout=0, dropouth=0, dropouti=0, n_input=40, n_output=139):
        super(MyModel, self).__init__()
        self.nhid = nhid
        self.nlayers = nlayers
        self.lockdrop = LockedDropout()
        self.dropout = dropout
        self.dropouth = dropouth
        self.dropouti = dropouti


        self.cnns = torch.nn.Sequential(
            nn.BatchNorm1d(n_input),  # normalize input channels

            nn.Conv1d(n_input, nhid, 3, padding=1),
            nn.BatchNorm1d(nhid),
            nn.Hardtanh(inplace=True),

            # nn.Conv1d(nhid, nhid, 3, padding=1),
            # nn.BatchNorm1d(nhid),
            # nn.LeakyReLU(inplace=True),
            #
            # nn.Conv1d(nhid, nhid, 3, padding=1),
            # nn.BatchNorm1d(nhid),
            # nn.LeakyReLU(inplace=True),
        )

        rnns = [BnLSTM(nhid, nhid)]
        for i in range(nlayers - 1):
            rnns.append(BnLSTM(nhid * 2, nhid))

        # rnns = [nn.LSTM(nhid, nhid, bidirectional=True)]
        # for i in range(nlayers - 1):
        #     rnns.append(nn.LSTM(nhid * 2, nhid, bidirectional=True))

        self.rnns = torch.nn.ModuleList(rnns)

        self.linear_out = nn.Linear(nhid * 2, n_output)
        print(self.rnns)

        self.init_weights()
        self.hs = []

    def forward(self, data, seq_sizes):

        data = data.permute(1, 2, 0)
        data = data.contiguous()
        # data: (batch, channels, seq_len)
        out = self.cnns(data)
        out = out.permute(2, 0, 1)

        # out: (seq_len, batch, channels)
        out = self.lockdrop(out, self.dropouti)

        # out = pack_padded_sequence(out, seq_sizes)

        for l, rnn in enumerate(self.rnns):
            # out, h = rnn(out, self.hs[l])
            # out, h = rnn(out)
            out, h = rnn(out, seq_sizes) #BnLSTM
            out = self.lockdrop(out, self.dropout if l == self.nlayers - 1 else self.dropouth)

        # out, _ = pad_packed_sequence(out)

        out = self.linear_out(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                print("init", m)
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                print("init", m)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                print("init", m)
                # nn.init.xavier_normal(m.weight.data)
                # m.weight.data.uniform_(-0.1, 0.1)
                m.bias.data.zero_()
        # Special bias init for last linear layer
        self.linear_out.bias.data = to_float_tensor(PHONEME_DISTRIBUTION)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        self.hs = []
        for l in range(self.nlayers):
            n = self.nhid
            self.hs.append((Variable(weight.new(1, bsz, n).zero_()),
                            Variable(weight.new(1, bsz, n).zero_())))

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)


def cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2)


def label2str(label):
    return "".join(PHONEME_MAP[i] for i in label)


def decode(output, seq_sizes, beam_width=40):
    decoder = CTCBeamDecoder(labels=PHONEME_MAP, blank_id=0, beam_width=beam_width)
    output = torch.transpose(output, 0, 1)  # batch, seq_len, probs
    probs = F.softmax(output, dim=2).data.cpu()

    output, scores, timesteps, out_seq_len = decoder.decode(probs=probs,
                                                            seq_lens=torch.IntTensor(seq_sizes))
    #     print("output", output)
    #     print("scores", scores)
    #     print("timesteps", timesteps)
    #     print("out_seq_len", out_seq_len)
    decoded = []
    for i in range(output.size(0)):
        chrs = ""
        if out_seq_len[i, 0] != 0:
            chrs = "".join(PHONEME_MAP[o] for o in output[i, 0, :out_seq_len[i, 0]])
        decoded.append(chrs)
    return decoded


def print_score_board(score_board, N=10):
    # sort by err asc
    score_board.sort(key=lambda x: x[0])
    print("\nTop-{}\n".format(N))
    print("\n".join(["CER:\t{:6.4f}\nLabel:\t{}\nMine:\t{}\n".format(i[0], i[1], i[2]) for i in score_board[:N]]))
    print("\nLast-{}\n".format(N))
    print("\n".join(["CER:\t{:6.4f}\nLabel:\t{}\nMine:\t{}\n".format(i[0], i[1], i[2]) for i in score_board[-N:]]))


def evaluate(model, criterion, loader, args, calc_cer=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    total_cer = 0
    score_board = []
    for batch, (frames, seq_sizes, labels, label_sizes) in enumerate(loader):
        batch += 1

        if args.cuda:
            frames = frames.cuda()
        frames = Variable(frames, volatile=True)
        output = model(frames, seq_sizes)

        loss = criterion(output,
                         Variable(torch.cat(labels).int(), volatile=True),
                         Variable(torch.IntTensor(seq_sizes), volatile=True),
                         Variable(label_sizes, volatile=True))
        total_loss += to_np(loss)[0]

        # if (batch == 1) or calc_cer:
        #     decoded = decode(output, seq_sizes)
        #     label_str = [label2str(label) for label in labels]
        #     for l, m in zip(label_str, decoded):
        #         e = cer(l, m)
        #         total_cer += e
        #         score_board.append([e * 100.0 / len(l), l, m])
            # if batch == 1:
            #     for l, m in zip(label_str[:10], decoded[:10]):
            #         print("Label:\t{}\nMine:\t{}\n".format(l, m))

    # print_score_board(score_board)

    total_phonemes = loader.dataset.total_phonemes
    return total_loss / (len(loader) * args.batch_size), total_loss / total_phonemes, total_cer * 100.0 / total_phonemes


def train(epoch, model, optimizer, criterion, loader, args):
    # Turn on training mode which enables dropout.
    model.train()
    sum_loss, sum_phonemes = 0, 0
    start_time = time.time()
    optimizer.zero_grad()

    for batch, (frames, seq_sizes, labels, label_sizes) in enumerate(loader):
        batch += 1

        # frames: (batch, seq_len, channels)
        # labels: (batch, n_phonemes)
        #         print("frames.size()", frames.size())
        #         print("labels.size()", labels.size())

        sum_phonemes += sum(len(l) for l in labels)

        if args.cuda:
            frames = frames.cuda()
        frames = Variable(frames)

        #         print("frames", frames)
        output = model(frames, seq_sizes)

        loss = criterion(output,
                         Variable(torch.cat(labels).int()),
                         Variable(torch.IntTensor(seq_sizes)),
                         Variable(label_sizes))
        loss.backward()

        sum_loss += loss.data[0]

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        #         print("parameter mean", np.mean([p.data.mean() for p in model.parameters()]))
        optimizer.step()
        optimizer.zero_grad()

        if batch % args.log_interval == 0:
            #             decoded = decode(output, seq_sizes)
            #             label_str = [label2str(label) for label in labels]
            #             for l, m in zip(label_str, decoded):
            #                 print("Label:\t{}\nMine:\t{}\n".format(l, m))
            #                 break

            elapsed = time.time() - start_time
            avg_loss = sum_loss / sum_phonemes
            print(
                '| epoch {:3d} | {:5d}/{:5d} batches ({:5.2f}%) | lr {:.2e} | {:3.0f} ms/utter | loss/utter {:5.2f} | loss/phoneme {:5.4f}'
                .format(epoch, batch, len(loader), (100.0 * batch / len(loader)),
                        optimizer.param_groups[0]['lr'],
                        elapsed * 1000.0 / (args.log_interval * args.batch_size),
                        sum_loss / (args.log_interval * args.batch_size),
                        avg_loss))
            sum_loss, sum_phonemes = 0, 0
            start_time = time.time()


def main(args):
    print('Args:', args)

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = MyModel(args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti)
    # model = MyModel(args.nhid, args.nlayers)
    # model.load_state_dict(torch.load("/mnt/part2/e1/010_0.00.w"))
    if args.cuda:
        model.cuda()

    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
    print('Model total parameters:', total_params)

    train_loader, dev_loader = get_data_loaders(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    criterion = CTCLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=0.01, verbose=True)
    for epoch in range(1, args.epochs + 1):
        print(datetime.now())
        epoch_start_time = time.time()
        train(epoch, model, optimizer, criterion, train_loader, args)
        if True:
            val_loss_utter, val_loss_phoneme, val_cer = evaluate(model, criterion, dev_loader, args, calc_cer=True)
            scheduler.step(val_loss_phoneme)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | loss/utter {:5.2f} | loss/phoneme {:5.4f} | valid cer {:5.4f}'
                  .format(epoch, (time.time() - epoch_start_time),
                          val_loss_utter, val_loss_phoneme, val_cer))
            print('-' * 89)

            if not os.path.exists(args.weights_dir):
                os.makedirs(args.weights_dir)
            weight_fname = "{}/{:03d}_{}.w".format(args.weights_dir, epoch, "{:.2f}".format(val_loss_phoneme))
            print("saving as", weight_fname)
            torch.save(model.state_dict(), weight_fname)


def get_test_data_loaders(args):
    print("loading data")
    xtest = np.load(args.data_dir + '/test.npy')
    print("load complete")
    kwargs = {'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        MyDataset(xtest, None),
        batch_size=1, shuffle=False, collate_fn=my_collate, **kwargs)
    return test_loader


def predict(args, out_fpath, weights_fpath):
    model = MyModel(args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti)
    # model = MyModel(args.nhid, args.nlayers)
    model.load_state_dict(torch.load(weights_fpath))
    if args.cuda:
        model.cuda()

    model.eval()
    print("model.training:", model.training)
    test_loader = get_test_data_loaders(args)

    results = []
    for batch, (frames, seq_sizes, _, _) in enumerate(test_loader):
        batch += 1
        if batch % args.log_interval == 0:
            print("utter", batch)

        if args.cuda:
            frames = frames.cuda()
        frames = Variable(frames, volatile=True)
        output = model(frames, seq_sizes)
        output = to_np(output)
        results.append(output[:, 0, 1:])

    np.save(out_fpath, results)
    print("done")


if __name__ == "__main__":
    print(__file__)
    with open(__file__) as f:
        print(f.read())
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and args.cuda
    main(args)
    # predict(args, "submission.csv", "004_0.12.w")
