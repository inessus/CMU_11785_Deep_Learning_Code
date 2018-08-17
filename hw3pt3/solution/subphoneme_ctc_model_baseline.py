import argparse
import os
import sys

import numpy as np
import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _use_shared_memory
from warpctc_pytorch import CTCLoss

from phoneme_list import N_STATES

INPUT_DIM = 40


def load_data(name, test=False):
    # Load the numpy files
    path = '../../dataset'
    features = np.load(os.path.join(path, '{}.npy'.format(name)))
    if test:
        labels = np.array([np.zeros((1,), dtype=np.int64) for _ in range(features.shape[0])])
    else:
        labels = np.load(os.path.join(path, '{}_subphonemes.npy'.format(name)))
    return features, labels


class SpeechDataset(Dataset):
    # Simple custom dataset
    def __init__(self, features, labels):
        self.features = [torch.from_numpy(x) for x in features]
        self.labels = [torch.from_numpy(x) for x in labels]
        assert len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]


def speech_collate_fn(batch):
    # Custom collation function
    batch.sort(key=lambda x: x[0].size(0), reverse=True)
    n = len(batch)

    if _use_shared_memory:
        utterance_lengths = torch.IntStorage._new_shared(n).new(n)
        label_lengths = torch.IntStorage._new_shared(n).new(n)
    else:
        utterance_lengths = torch.IntTensor(n)
        label_lengths = torch.IntTensor(n)

    for i, (x, y) in enumerate(batch):
        utterance_lengths[i] = x.size(0)
        label_lengths[i] = y.size(0)

    umax = utterance_lengths.max()
    ltot = label_lengths.sum()

    if _use_shared_memory:
        utterances = torch.FloatStorage._new_shared(umax * n * INPUT_DIM).new(umax, n, INPUT_DIM).zero_()
        labels = torch.IntStorage._new_shared(ltot).new(ltot)
    else:
        utterances = torch.FloatTensor(umax, n, INPUT_DIM).zero_()
        labels = torch.IntTensor(ltot)

    lpos = 0
    for i, (x, y) in enumerate(batch):
        utterances[:x.size(0), i, :] = x
        llen = y.size(0)
        labels[lpos:lpos + llen] = y
        lpos += llen
    assert lpos == ltot
    return utterances, utterance_lengths, label_lengths, labels


def make_loader(name, args, shuffle=True, batch_size=64, test=False):
    # Build the data loader
    kwargs = {'pin_memory': True, 'num_workers': args.num_workers} if args.cuda else {}
    data = load_data(name, test=test)
    dataset = SpeechDataset(*data)
    loader = DataLoader(dataset, collate_fn=speech_collate_fn, shuffle=shuffle, batch_size=batch_size, **kwargs)
    return loader


class AdvancedLSTM(nn.LSTM):
    # Custom LSTM class for learing initial states
    def __init__(self, *args, **kwargs):
        super(AdvancedLSTM, self).__init__(*args, **kwargs)
        bi = 2 if self.bidirectional else 1
        self.stateh0 = nn.Parameter(torch.FloatTensor(bi, 1, self.hidden_size).zero_())
        self.statec0 = nn.Parameter(torch.FloatTensor(bi, 1, self.hidden_size).zero_())

    def forward(self, input):
        n = input.batch_sizes[0]
        return super(AdvancedLSTM, self).forward(
            input,
            hx=(
                self.stateh0.expand(-1, n, -1).contiguous(),
                self.statec0.expand(-1, n, -1).contiguous()))


class SpeechModel(nn.Module):
    # Model is 3 LSTM and then a linear layer
    def __init__(self, args):
        super(SpeechModel, self).__init__()
        self.rnns = nn.ModuleList()
        self.rnns.append(AdvancedLSTM(input_size=INPUT_DIM, hidden_size=args.hidden_dim, bidirectional=True))
        self.rnns.append(AdvancedLSTM(input_size=args.hidden_dim * 2, hidden_size=args.hidden_dim, bidirectional=True))
        self.rnns.append(AdvancedLSTM(input_size=args.hidden_dim * 2, hidden_size=args.hidden_dim, bidirectional=True))
        self.output_layer = nn.Linear(in_features=args.hidden_dim * 2, out_features=N_STATES + 1)
        self.lsm = nn.LogSoftmax(dim=2)

    def forward(self, features, feature_lengths, label_lengths):
        h = pack_padded_sequence(features, feature_lengths.data.cpu().numpy())
        for l in self.rnns:
            h, _ = l(h)
        h, _ = pad_packed_sequence(h)
        logits = self.output_layer(h)  # (t, n, l)
        logits = self.lsm(logits)
        return logits, feature_lengths, label_lengths


def run_logits(loader, model, args):
    # Save the logits into a numpy file
    sens = []
    for features, feature_lengths, label_lengths, labels in loader:
        if args.cuda:
            features = features.cuda()
        features = Variable(features)
        logits, _a, _b = model(features, Variable(feature_lengths), label_lengths)
        print("Max 0: {}, mean: {}".format(logits[:, :, 0].max().cpu().data.numpy(),
                                           logits[:, :, 0].mean().cpu().data.numpy()))
        logits = logits[:, :, 1:].cpu().data.numpy()
        for i in range(logits.shape[1]):
            size = feature_lengths[i]
            x = logits[:size, i, :]
            sens.append(x)
    sens = np.array(sens)
    np.save(os.path.join(args.save_directory, 'sens.npy'), sens)


class CTCCriterion(CTCLoss):
    # Custom CTC for use with inferno
    def forward(self, prediction, target):
        assert len(prediction) == 3
        acts = prediction[0]
        act_lens = prediction[1].int()
        label_lens = prediction[2].int()
        labels = target + 1
        ctcloss = super(CTCCriterion, self).forward(
            acts=acts,
            labels=labels.cpu(),
            act_lens=act_lens.cpu(),
            label_lens=label_lens.cpu()
        )
        print("CTCLoss: {}".format(ctcloss.cpu().data[0]))
        return ctcloss


def run(args):
    model = SpeechModel(args)
    trainer = Trainer()
    if os.path.exists(os.path.join(args.save_directory, trainer._checkpoint_filename)):
        trainer.load(from_directory=args.save_directory)
        model.load_state_dict(trainer.model.state_dict())
        if args.cuda:
            model = model.cuda()
    else:
        train_loader = make_loader('train', args, batch_size=args.batch_size)
        dev_loader = make_loader('dev', args, batch_size=args.batch_size)
        # Build trainer
        trainer = Trainer(model) \
            .build_criterion(CTCCriterion, size_average=True) \
            .build_optimizer('Adam', weight_decay=1e-7, lr=5e-5) \
            .validate_every((1, 'epochs')) \
            .save_every((1, 'epochs')) \
            .save_to_directory(args.save_directory) \
            .set_max_num_epochs(args.epochs) \
            .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                            log_images_every='never'),
                          log_directory=args.save_directory)

        # Bind loaders
        trainer.bind_loader('train', train_loader, num_inputs=3, num_targets=1)
        trainer.bind_loader('validate', dev_loader, num_inputs=3, num_targets=1)

        if args.cuda:
            trainer.cuda()

        # Go!
        trainer.fit()
        trainer.save()

    test_loader = make_loader('test', args=args, shuffle=False, batch_size=1, test=True)
    run_logits(loader=test_loader, model=model, args=args)


def main(argv):
    # Training settings
    parser = argparse.ArgumentParser(description='Homework 3 Part 3')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--save-directory', type=str, default='output/hw3p3baseline/v1',
                        help='output directory')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--num-workers', type=int, default=2, metavar='N', help='number of workers')
    parser.add_argument('--hidden-dim', type=int, default=256, metavar='N',
                        help='hidden dim')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
