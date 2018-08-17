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
import torch.optim.lr_scheduler
from torchvision import datasets, transforms
from torch.autograd import Variable

import Levenshtein as Lev

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('--data-dir', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--weights-dir', type=str, default='./weights/',
                    help='data directory')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, #default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=2222,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval')
parser.add_argument('--wdecay', type=float, default=1.2e-6, #default=1.2e-6,
                    help='weight decay applied to all weights')
# args = parser.parse_args("")
# args.cuda = torch.cuda.is_available() and args.cuda


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


idx2chr = ['<eos>', ' ', "'", '+', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']
chr2idx = {'<eos>': 0, ' ': 1, "'": 2, '+': 3, '-': 4, '.': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, 'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23, 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, '_': 32}

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        if y is not None:
            self.total_labels = sum(len(yi) for yi in y)
        else:
            self.total_labels = -1

        print("n_utters", self.x.shape[0], "total_labels", self.total_labels)

    def __getitem__(self, idx):
        frames = self.x[idx]
        if self.y is None:
            labels = [-1]
        else:
            labels = [chr2idx[c] for c in self.y[idx]]
            labels = labels + [chr2idx['<eos>']]
        return to_float_tensor(frames), \
               to_int_tensor(np.array(labels))

    def __len__(self):
        return self.x.shape[0]  # // 10


def my_collate(batch):
    batch_size = len(batch)
    batch = sorted(batch, key=lambda b: b[0].size(0), reverse=True)  # sort the batch by seq_len desc
    max_seq_len = batch[0][0].size(0)
    channels = batch[0][0].size(1)
    pack = torch.zeros(max_seq_len, batch_size, channels)
    seq_sizes = []
    max_label_len = max(label.size(0) for (f, label) in batch)
    all_labels = torch.zeros(batch_size, max_label_len).long()
    label_sizes = torch.zeros(batch_size).int()
    for i, (frames, label) in enumerate(batch):
        seq_size = frames.size(0)
        seq_sizes.append(seq_size)

        labele_size = label.size(0)
        label_sizes[i] = labele_size

        pack[:seq_size, i, :] = frames
        all_labels[i, :labele_size] = label
    return pack, seq_sizes, all_labels, label_sizes


def get_data_loaders(args, use_dev_for_train=False):
    print("loading data")

    xdev = np.load(args.data_dir + '/dev.npy')
    ydev = np.load(args.data_dir + '/dev_transcripts.npy')
    if use_dev_for_train:
        xtrain = np.load(args.data_dir + '/dev.npy')
        ytrain = np.load(args.data_dir + '/dev_transcripts.npy')
    else:
        xtrain = np.load(args.data_dir + '/train.npy')
        ytrain = np.load(args.data_dir + '/train_transcripts.npy')

    print("load complete")
    kwargs = {'num_workers': 3, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        MyDataset(xtrain, ytrain),
        batch_size=args.batch_size, shuffle=True, collate_fn=my_collate, **kwargs)
    dev_loader = torch.utils.data.DataLoader(
        MyDataset(xdev, ydev),
        batch_size=args.batch_size, shuffle=True, collate_fn=my_collate, **kwargs)

    return train_loader, dev_loader


class pBLSTMLayer(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, rnn_unit='LSTM', dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit.upper())
        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim * 2, hidden_dim, 1, bidirectional=True,
                                   dropout=dropout_rate, batch_first=True)

    # BLSTM layer for pBLSTM
    # Step 1. Reduce time resolution to half
    # Step 2. Run through BLSTM
    def forward(self, input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        # make input len even number
        if timestep % 2 != 0:
            input_x = input_x[:, :-1, :]
            timestep -= 1
        feature_dim = input_x.size(2)
        # Reduce time resolution
        input_x = input_x.contiguous().view(batch_size, timestep // 2, feature_dim * 2)
        # Bidirectional RNN
        output, hidden = self.BLSTM(input_x)
        return output, hidden

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        # x': (B, L, C)
        if dropout == 0 or not self.training:
            return x
        mask = x.data.new(x.size(0), 1, x.size(2))
        mask = mask.bernoulli_(1 - dropout)
        mask = Variable(mask, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

# Listener is a pBLSTM stacking 3 layers to reduce time resolution 8 times
# Input shape should be [batch_size, max_seq_len, channels]
class Listener(nn.Module):
    def __init__(self, input_feature_dim, listener_hidden_dim, dropout_rate=0.0, dropout=0, dropouth=0, dropouti=0):
        super(Listener, self).__init__()
        self.cnns = torch.nn.Sequential(
            nn.BatchNorm1d(input_feature_dim),  # normalize input channels
            #             nn.Conv1d(input_feature_dim, nhid, 3, padding=1),
            #             nn.BatchNorm1d(nhid),
            #             nn.Hardtanh(inplace=True),
        )
        # Listener RNN layer
        self.rnn1 = pBLSTMLayer(input_feature_dim, listener_hidden_dim,
                                dropout_rate=dropout_rate)
        self.rnn2 = pBLSTMLayer(listener_hidden_dim * 2, listener_hidden_dim,
                                dropout_rate=dropout_rate)
        self.rnn3 = pBLSTMLayer(listener_hidden_dim * 2, listener_hidden_dim,
                                dropout_rate=dropout_rate)
        self.lockdrop = LockedDropout()
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropout = dropout

    def forward(self, frames, seq_sizes):
        # frames: (max_seq_len, batch_size, channels)

        frames = frames.permute(1, 2, 0).contiguous()
        # frames: (batch, channels, seq_len)
        frames = self.cnns(frames)

        output = frames.permute(0, 2, 1)
        # output: (batch_size, max_seq_len, channels)

        output = self.lockdrop(output, self.dropouti)
        output, _ = self.rnn1(output)
        output = self.lockdrop(output, self.dropouth)
        output, _ = self.rnn2(output)
        output = self.lockdrop(output, self.dropouth)
        output, _ = self.rnn3(output)
        output = self.lockdrop(output, self.dropout)

        # shorten for 8x
        out_seq_sizes = [size // 8 for size in seq_sizes]

        return output, out_seq_sizes


# Speller specified in the paper
class Speller(nn.Module):
    def __init__(self, n_classes, speller_hidden_dim, speller_rnn_layer, attention, context_dim):
        super(Speller, self).__init__()
        self.n_classes = n_classes
        self.rnn_unit = nn.LSTMCell

        # self.rnn_layer = self.rnn_unit(output_class_dim + speller_hidden_dim, speller_hidden_dim,
        #                                num_layers=speller_rnn_layer)

        self.rnn_layer = torch.nn.ModuleList()
        self.rnn_inith = torch.nn.ParameterList()
        self.rnn_initc = torch.nn.ParameterList()
        self.rnn_layer.append(self.rnn_unit(speller_hidden_dim + context_dim, speller_hidden_dim))
        for i in range(speller_rnn_layer):
            if i != 0:
                self.rnn_layer.append(self.rnn_unit(speller_hidden_dim, speller_hidden_dim))
            self.rnn_inith.append(torch.nn.Parameter(torch.rand(1, speller_hidden_dim)))
            self.rnn_initc.append(torch.nn.Parameter(torch.rand(1, speller_hidden_dim)))

        self.attention = attention

        # char embedding
        self.embed = nn.Embedding(n_classes, speller_hidden_dim)

        # prob output layers
        self.fc = nn.Linear(speller_hidden_dim + context_dim, speller_hidden_dim)
        self.activate = torch.nn.LeakyReLU(negative_slope=0.2)
        self.unembed = nn.Linear(speller_hidden_dim, n_classes)
        self.unembed.weight = self.embed.weight
        self.character_distribution = nn.Sequential(self.fc, self.activate, self.unembed)

    def forward(self, listener_feature, seq_sizes, max_iters, ground_truth=None, teacher_force_rate=0.9, dropout=[]):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False
        batch_size = listener_feature.size()[0]

        state, output_word = self.get_initial_state(batch_size)

        # dropouts
        dropout_masks = []
        if dropout and self.training:
            h = state[0][0] # B, C
            n_layers = len(state[0])
            for i in range(n_layers):
                mask = h.data.new(h.size(0), h.size(1)).bernoulli_(1 - dropout[i]) / (1 - dropout[i])
                dropout_masks.append(Variable(mask, requires_grad=False))

        raw_pred_seq = []
        attention_record = []
        for step in range(ground_truth.size(1) if ground_truth is not None else max_iters):

            # print("last_output_word_forward", idx2chr[output_word.data[0]])
            attention_score, raw_pred, state = self.run_one_step(listener_feature, seq_sizes, output_word, state, dropout_masks=dropout_masks)

            attention_record.append(attention_score)
            raw_pred_seq.append(raw_pred)

            # Teacher force - use ground truth as next step's input
            if teacher_force:
                output_word = ground_truth[:, step]
            else:
                output_word = torch.max(raw_pred, dim=1)[1]

        return torch.stack(raw_pred_seq, dim=1), attention_record

    def run_one_step(self, listener_feature, seq_sizes, last_output_word, state, dropout_masks=None):
        output_word_emb = self.embed(last_output_word)

        # get attention context
        hidden, cell = state[0], state[1]
        last_rnn_output = hidden[-1]  # last layer
        # print("last_rnn_output", last_rnn_output)
        # print("listener_feature", listener_feature)
        attention_score, context = self.attention(last_rnn_output, listener_feature, seq_sizes)

        # run speller rnns for one time step
        rnn_input = torch.cat([output_word_emb, context], dim=1)
        new_hidden, new_cell = [None] * len(self.rnn_layer), [None] * len(self.rnn_layer)
        for l, rnn in enumerate(self.rnn_layer):
            new_hidden[l], new_cell[l] = rnn(rnn_input, (hidden[l], cell[l]))
            if dropout_masks:
                rnn_input = new_hidden[l] * dropout_masks[l]
            else:
                rnn_input = new_hidden[l]
        rnn_output = new_hidden[-1]  # last layer

        # make prediction
        concat_feature = torch.cat([rnn_output, context], dim=1)
        #             print("concat_feature.size()", concat_feature.size())
        raw_pred = self.character_distribution(concat_feature)
        #             print("raw_pred.size()", raw_pred.size())
        return attention_score, raw_pred, [new_hidden, new_cell]

    def get_initial_state(self, batch_size=32):
        hidden = [h.repeat(batch_size, 1) for h in self.rnn_inith]
        cell = [c.repeat(batch_size, 1) for c in self.rnn_initc]
        # <sos> (same vocab as <eos>)
        output_word = Variable(hidden[0].data.new(batch_size).long().fill_(chr2idx['<eos>']))
        return [hidden, cell], output_word


def SequenceWise(input_module, input_x):
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size, time_steps, -1)


class Attention(nn.Module):
    def __init__(self, key_query_dim=128, speller_query_dim=256, listener_feature_dim=512, context_dim=128):
        # context_dim: C, key_query_dim: C'
        super(Attention, self).__init__()
        self.softmax = nn.Softmax()
        self.fc_query = nn.Linear(speller_query_dim, key_query_dim)
        self.fc_key = nn.Linear(listener_feature_dim, key_query_dim)
        self.fc_value = nn.Linear(listener_feature_dim, context_dim)
        self.activate = torch.nn.LeakyReLU(negative_slope=0.2)

    def forward(self, decoder_state, listener_feature, seq_sizes):
        # listener_feature: B, L, C
        # print("listener_feature.size()", listener_feature.size())

        query = self.activate(self.fc_query(decoder_state))
        key = self.activate(SequenceWise(self.fc_key, listener_feature))

        # query: B, 1, C'
        # key  : B, L, C'

        # print("query.size()", query.size())
        # print("key.size()", key.size())
        energy = torch.bmm(query.unsqueeze(1), key.transpose(1, 2)).squeeze(dim=1)

        # energy/attention_score: B, L
        # masked softmax
        mask = Variable(energy.data.new(energy.size(0), energy.size(1)).zero_(), requires_grad=False)
        for i, size in enumerate(seq_sizes):
            mask[i, :size] = 1
        attention_score = self.softmax(energy)
        attention_score = mask * attention_score
        attention_score = attention_score / torch.sum(attention_score, dim=1).unsqueeze(1).expand_as(attention_score)

        # value: B, L, C
        value = self.activate(self.fc_value(listener_feature))
        context = torch.bmm(attention_score.unsqueeze(1), value).squeeze(dim=1)

        # context: B, C
        return attention_score, context


class LAS(nn.Module):
    def __init__(self):
        super(LAS, self).__init__()
        # self.listener = Listener(40, 256)
        self.listener = Listener(40, 256, dropouti=0.3, dropouth=0.1, dropout=0.2)
        self.attention = Attention(key_query_dim=128, speller_query_dim=256, listener_feature_dim=512, context_dim=128)
        self.speller = Speller(33, 256, 3, self.attention, context_dim=128)

        # for generation
        self._listener_feature = None
        self._seq_sizes = None
        self.softmax = nn.Softmax()

    def forward(self, frames, seq_sizes, labels, max_iters=250):
        """
        :param frames: (batch_size, max_seq_len, channels)
        :param seq_sizes:
        :return: outputs for all batchs
        """
        listener_features, out_seq_sizes = self.listener(frames, seq_sizes)
        outputs, attentions = self.speller(listener_features, out_seq_sizes, max_iters, labels,
                                           teacher_force_rate=0.8, dropout=[0.1, 0.1, 0.3])
        return outputs, attentions

    def get_initial_state(self, frames, seq_sizes):
        self._listener_feature, self._seq_sizes = self.listener(frames, seq_sizes)
        state, output_word = self.speller.get_initial_state(batch_size=1)
        return state, output_word.data[0]

    def generate(self, frames, last_output_words, last_states):
        """
        :param frames:
        :param last_output_words: List of indexes in vocab
        :param last_states:
        :return:
        """
        new_states, raw_preds, attention_scores = [], [], []
        for last_output_word, last_state in zip(last_output_words, last_states):
            # print("last_state", last_state)
            # print("last_output_word", idx2chr[last_output_word])
            last_output_word = Variable(self._listener_feature.data.new(1).long().fill_(int(last_output_word)))
            attention_score, raw_pred, new_state = \
                self.speller.run_one_step(self._listener_feature, self._seq_sizes, last_output_word, last_state)
            new_states      .append(new_state)
            raw_preds       .append(self.softmax(raw_pred).squeeze().data.cpu().numpy())
            attention_scores.append(attention_score)

        return new_states, raw_preds, attention_scores


def cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2)

def label_list_to_str(labels):
    output = []
    for l in labels:
        output.append("".join(idx2chr[i] for i in l))
    return output

def labels2str(labels, label_sizes):
    output = []
    for l, s in zip(labels, label_sizes):
        output.append("".join(idx2chr[i] for i in l[:s]))
    return output


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
            frames, labels = frames.cuda(), labels.cuda()
        frames, labels = Variable(frames, volatile=True), Variable(labels, volatile=True)

        outputs, attentions = model(frames, seq_sizes, labels)
        loss = criterion(outputs, label_sizes, labels)

        total_loss += to_np(loss)[0]

        if (batch == 1) or calc_cer:
            decoded = greedy_decode(F.softmax(outputs).data.cpu())
            labels_str = labels2str(to_np(labels), label_sizes)  # [label2str(label) for label in to_np(labels)]
            for l, m in zip(labels_str, decoded):
                e = cer(l, m)
                total_cer += e
                score_board.append([e * 100.0 / len(l), l, m])
                # if batch == 1:
                #     for l, m in zip(label_str[:10], decoded[:10]):
                #         print("Label:\t{}\nMine:\t{}\n".format(l, m))

    print_score_board(score_board)
    total_labels = loader.dataset.total_labels
    return total_loss / (len(loader) * args.batch_size), total_loss / total_labels, total_cer * 100.0 / total_labels


def train(epoch, model, optimizer, criterion, loader, args):
    # Turn on training mode which enables dropout.
    model.train()
    sum_loss, sum_labels = 0, 0
    start_time = time.time()
    optimizer.zero_grad()

    for batch, (frames, seq_sizes, labels, label_sizes) in enumerate(loader):
        batch += 1

        # frames: (max_seq_len, batch_size, channels)
        # labels: (batch, individual_label_len)
        #         print("frames.size()", frames.size())
        #         print("labels.size()", labels.size())


        sum_labels += sum(label_sizes)

        if args.cuda:
            frames, labels = frames.cuda(), labels.cuda()
        frames, labels = Variable(frames), Variable(labels)

        outputs, attentions = model(frames, seq_sizes, labels)

        loss = criterion(outputs, label_sizes, labels)
        loss.backward()

        sum_loss += loss.data[0]

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        optimizer.step()
        optimizer.zero_grad()

        if batch % args.log_interval == 0:
            #             decoded = decode(output, seq_sizes)
            #             label_str = [label2str(label) for label in labels]
            #             for l, m in zip(label_str, decoded):
            #                 print("Label:\t{}\nMine:\t{}\n".format(l, m))
            #                 break

            elapsed = time.time() - start_time
            avg_loss = sum_loss / sum_labels
            print(
                '| epoch {:3d} | {:5d}/{:5d} batches ({:5.2f}%) | lr {:.2e} | {:3.0f} ms/utter | loss/utter {:5.2f} | loss/label {:5.4f}'
                    .format(epoch, batch, len(loader), (100.0 * batch / len(loader)),
                            optimizer.param_groups[0]['lr'],
                            elapsed * 1000.0 / (args.log_interval * args.batch_size),
                            sum_loss / (args.log_interval * args.batch_size),
                            avg_loss))
            sum_loss, sum_labels = 0, 0
            start_time = time.time()


def find_first_eos_in_pred(pred):
    # pred: L, C
    chrs = pred.max(1)[1].data.cpu().numpy()
    # print("chrs", chrs)
    for idx, c in enumerate(chrs):
        if c == chr2idx['<eos>']:
            return idx
    return len(chrs)

class SeqCrossEntropyLoss(torch.nn.Module):
    """
    Edit-Distance-Aware Seq2Seq loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, preds, label_sizes, labels):
        # preds: B, L, C
        pred_list = []
        label_list = []
        max_iter = preds.size(1)
        for (pred, label, num_iter) in zip(preds, labels, label_sizes):
            pred_for_loss = []
            label_for_loss = []
            eos_idx = find_first_eos_in_pred(pred)

            # print("before pred ", pred[:num_iter].max(1)[1].data.cpu().numpy())
            # print("before label", label[:num_iter].data.cpu().numpy())
            if eos_idx < num_iter:
                # early-stopping penalty
                # label: a b c d e f g h i $
                # pred : a b c d $
                # will turn pred to
                #        a b c d $ $ $ $ $ $

                # print("< eos ", eos_idx, "label eos", num_iter)
                if eos_idx != 0:
                    pred_for_loss.append(pred[:eos_idx])
                pred_for_loss += [pred[eos_idx:eos_idx + 1]] * (num_iter - eos_idx)
                label_for_loss.append(label[:num_iter])

            elif eos_idx == max_iter:
                # non-stopping pentalty:
                # label: a b c d $
                # pred : a b c d e f g h i j
                # will turn label to
                #        a b c d $ $ $ $ $ $

                # print("$ eos", eos_idx, "label eos", num_iter)
                pred_for_loss.append(pred[:eos_idx])
                label_for_loss.append(label[:num_iter])
                label_for_loss += [label[num_iter - 1:num_iter]] * (eos_idx - num_iter)

            else:
                # stopping at max_iter pentalty:
                # label: a b c d $
                # pred : a b c d e f g h i $
                # will turn label to
                #        a b c d $ $ $ $ $ $

                # print("> eos", eos_idx, "label eos", num_iter)
                pred_for_loss.append(pred[:eos_idx + 1])
                label_for_loss.append(label[:num_iter])
                label_for_loss += [label[num_iter - 1:num_iter]] * (eos_idx + 1 - num_iter)

            # print("after  pred", torch.cat(pred_for_loss).max(1)[1].data.cpu().numpy())
            # print("after  label", torch.cat(label_for_loss).data.cpu().numpy())

            pred_list.append(torch.cat(pred_for_loss))
            label_list.append(torch.cat(label_for_loss))

        preds_batch = torch.cat(pred_list)
        labels_batch = torch.cat(label_list)
        loss = torch.nn.functional.cross_entropy(preds_batch, labels_batch, size_average=False)
        return loss


def main(args):
    print('Args:', args)

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = LAS()
    model.load_state_dict(torch.load("../e13/006_79.4130.w"))

    if args.cuda:
        model.cuda()

        # total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
        # print('Model total parameters:', total_params)

    train_loader, dev_loader = get_data_loaders(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    criterion = SeqCrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.01, verbose=True)
    for epoch in range(1, args.epochs + 1):
        print(datetime.now())
        epoch_start_time = time.time()
        train(epoch, model, optimizer, criterion, train_loader, args)
        if True:
            val_loss_utter, val_loss_label, val_cer = evaluate(model, criterion, dev_loader, args, calc_cer=True)
            scheduler.step(val_loss_label)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | loss/utter {:5.2f} | loss/label {:5.4f} | valid cer {:5.4f}'
                  .format(epoch, (time.time() - epoch_start_time),
                          val_loss_utter, val_loss_label, val_cer))
            print('-' * 89)

            if not os.path.exists(args.weights_dir):
                os.makedirs(args.weights_dir)
            weight_fname = "{}/{:03d}_{}.w".format(args.weights_dir, epoch, "{:.4f}".format(val_cer))
            print("saving as", weight_fname)
            torch.save(model.state_dict(), weight_fname)


def greedy_decode(probs):
    # probs: FloatTensor (B, L, C)
    out = []
    for prob in probs:
        s = []
        for step in prob:
            #             idx = torch.multinomial(step, 1)[0]
            idx = step.max(0)[1][0]
            c = idx2chr[idx]
            s.append(c)
            if c == '<eos>':
                break
        out.append("".join(s))
    return out


def get_test_data_loaders(args):
    print("loading data")
    xtest = np.load(args.data_dir + '/test.npy')
    print("load complete")
    kwargs = {'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        MyDataset(xtest, None),
        batch_size=1, shuffle=False, collate_fn=my_collate, **kwargs)
    return test_loader


def predict(args, csv_fpath, weights_fpath):
    model = LAS()
    model.load_state_dict(torch.load(weights_fpath))
    if args.cuda:
        model.cuda()

    model.eval()
    test_loader = get_test_data_loaders(args)
    # test_loader = dev_loader
    if True:
        #     with open(csv_fpath, 'w') as csvfile:
        #         writer = csv.DictWriter(csvfile, fieldnames=['Id', 'Predicted'])
        #         writer.writeheader()
        cnt = 0
        for batch, (frames, seq_sizes, _, _) in enumerate(test_loader):
            batch += 1
            if args.cuda:
                frames = frames.cuda()
            frames = Variable(frames, volatile=True)
            outputs, attentions = model(frames, None, max_iters=250)
            decoded = greedy_decode(F.softmax(outputs).data.cpu())
            for s in decoded:
                print("\n" + s + "\n")
            break
            #             for s in decoded:
            #                 if cnt % args.log_interval * 20 == 0:
            #                     print(cnt, s)
            #                 writer.writerow({"Id": cnt, "Predicted": s})
            #                 cnt += 1
    print("done")


if __name__ == "__main__":
    print(__file__)
    with open(__file__) as f:
        print(f.read())
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and args.cuda
    main(args)
    # predict(args, "submission.csv", "004_0.12.w")


